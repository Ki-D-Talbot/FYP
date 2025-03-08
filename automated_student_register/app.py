from flask import Flask, render_template, request, redirect, url_for, flash, session, Response, jsonify
from flask_sqlalchemy import SQLAlchemy
from flask_login import LoginManager, UserMixin, login_user, login_required, logout_user, current_user
import cv2
import datetime
import os
import numpy as np
import tensorflow as tf
from sklearn.metrics.pairwise import cosine_similarity

app = Flask(__name__)
app.secret_key = 'your_secret_key' 
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///database.db'
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False
db = SQLAlchemy(app)

# Initialize Flask-Login
login_manager = LoginManager()
login_manager.init_app(app)
login_manager.login_view = 'login'  # Redirect to login page if not logged in

class User(UserMixin, db.Model):
    id = db.Column(db.Integer, primary_key=True)
    username = db.Column(db.String(100), unique=True, nullable=False)
    password = db.Column(db.String(100), nullable=False)

class Student(db.Model):
    student_id = db.Column(db.Integer, primary_key=True)
    name = db.Column(db.String(100), nullable=False)
    email = db.Column(db.String(100), nullable=True)
    class_name = db.Column(db.String(50), nullable=True)
    photo_path = db.Column(db.String(200), nullable=True)

class Attendance(db.Model):
    attendance_id = db.Column(db.Integer, primary_key=True)
    student_id = db.Column(db.Integer, db.ForeignKey('student.student_id'), nullable=False)
    timestamp = db.Column(db.DateTime, default=db.func.current_timestamp())

# Ensure directory for student faces exists
os.makedirs('student_faces', exist_ok=True)

# Use app.app_context() to create tables
with app.app_context():
    db.create_all()  # Create tables

# Load facial recognition model
def load_model():
    try:
        model_path = 'facial_recognition_model.h5'
        model = tf.keras.models.load_model(model_path)
        print("Model loaded successfully")
        return model
    except Exception as e:
        print(f"Error loading model: {e}")
        return None

# Global variable for the model
facial_recognition_model = load_model()

@login_manager.user_loader
def load_user(user_id):
    return db.session.get(User, int(user_id))

@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        username = request.form['username']
        password = request.form['password']
        user = User.query.filter_by(username=username).first()
        if user and user.password == password:  # In a real app, use hashed passwords
            login_user(user)
            return redirect(url_for('admin_landing'))
        else:
            flash('Invalid username or password', 'danger')
    return render_template('login.html')

@app.route('/logout')
@login_required
def logout():
    logout_user()
    return redirect(url_for('login'))

@app.route('/')
@login_required
def admin_landing():
    return render_template('admin.html')

@app.route('/add_student', methods=['GET', 'POST'])
@login_required
def add_student():
    if request.method == 'POST':
        name = request.form['name']
        email = request.form.get('email')
        class_name = request.form.get('class_name')
        photo_path = request.form.get('photo_path')

        new_student = Student(name=name, email=email, class_name=class_name, photo_path=photo_path)
        db.session.add(new_student)
        db.session.commit()

        return redirect(url_for('list_students'))
    return render_template('add_student.html')

@app.route('/students')
@login_required
def list_students():
    students = Student.query.all()
    return render_template('students.html', students=students)

@app.route('/create_admin')
def create_admin():
    admin = User(username='admin', password='admin123')
    db.session.add(admin)
    db.session.commit()
    return "Admin user created!"

@app.route('/edit_student/<int:student_id>', methods=['GET', 'POST'])
@login_required
def edit_student(student_id):
    student = Student.query.get_or_404(student_id)
    
    if request.method == 'POST':
        student.name = request.form['name']
        student.email = request.form.get('email')
        student.class_name = request.form.get('class_name')
        student.photo_path = request.form.get('photo_path')

        db.session.commit()
        return redirect(url_for('list_students'))
    
    return render_template('edit_student.html', student=student)

@app.route('/capture_face/<int:student_id>')
@login_required
def capture_face(student_id):
    student = Student.query.get_or_404(student_id)
    return render_template('capture_face.html', student=student)

@app.route('/save_face', methods=['POST'])
@login_required
def save_face():
    if 'image' not in request.files:
        return jsonify({"error": "No image provided"}), 400
    
    image = request.files['image']
    student_id = request.form.get('student_id')
    
    if not student_id:
        return jsonify({"error": "No student ID provided"}), 400
    
    student_folder = os.path.join('student_faces', student_id)
    os.makedirs(student_folder, exist_ok=True)
    
    image_path = os.path.join(student_folder, 'face.jpg')
    try:
        image.save(image_path)
        
        student = Student.query.get(student_id)
        if student:
            student.photo_path = image_path
            db.session.commit()
        
        return jsonify({"success": True, "message": "Face saved successfully"}), 200
    except Exception as e:
        print(f"Error saving face: {e}")
        return jsonify({"error": str(e)}), 500

@app.route('/video_feed')
def video_feed():
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    camera = cv2.VideoCapture(0)
    
    # Load student face embeddings
    student_embeddings = {}
    if facial_recognition_model is not None:
        students = Student.query.all()
        for student in students:
            if student.photo_path and os.path.exists(student.photo_path):
                try:
                    img = cv2.imread(student.photo_path)
                    img = cv2.resize(img, (224, 224))  # Adjust based on your model
                    img = img.astype('float32') / 255.0
                    img = np.expand_dims(img, axis=0)
                    
                    embedding = facial_recognition_model.predict(img, verbose=0)
                    student_embeddings[student.student_id] = embedding
                    print(f"Loaded embedding for student: {student.name}")
                except Exception as e:
                    print(f"Error processing face for {student.name}: {e}")
    
    def generate_frames():
        frame_count = 0
        
        while True:
            success, frame = camera.read()
            if not success:
                print("Failed to capture frame")
                break
            
            frame_count += 1
            # Process every 3rd frame to reduce load
            process_recognition = (frame_count % 3 == 0) and facial_recognition_model is not None
            
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5)
            
            for (x, y, w, h) in faces:
                cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)
                
                if process_recognition and student_embeddings:
                    try:
                        face = frame[y:y+h, x:x+w]
                        face = cv2.resize(face, (224, 224))  # Adjust based on your model
                        face = face.astype('float32') / 255.0
                        face = np.expand_dims(face, axis=0)
                        
                        embedding = facial_recognition_model.predict(face, verbose=0)
                        
                        max_similarity = 0
                        recognized_student_id = None
                        recognized_name = "Unknown"
                        
                        for student_id, ref_embedding in student_embeddings.items():
                            similarity = cosine_similarity(embedding, ref_embedding)[0][0]
                            
                            if similarity > max_similarity and similarity > 0.7:  # Threshold
                                max_similarity = similarity
                                recognized_student_id = student_id
                                student = Student.query.get(student_id)
                                recognized_name = student.name if student else "Unknown"
                        
                        # Display name and confidence
                        label = f"{recognized_name} ({max_similarity:.2f})"
                        cv2.putText(frame, label, (x, y-10), 
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
                        
                        # Log attendance if recognized
                        if recognized_student_id is not None and max_similarity > 0.7:
                            log_attendance(recognized_student_id)
                    
                    except Exception as e:
                        print(f"Error processing face: {e}")
                else:
                    # If not using recognition model, use simple detection
                    log_attendance(1)  # Default to student_id 1
            
            ret, buffer = cv2.imencode('.jpg', frame)
            frame = buffer.tobytes()
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')
    
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

def log_attendance(student_id):
    # Prevent duplicate logs within short timeframe
    one_minute_ago = datetime.datetime.now() - datetime.timedelta(minutes=1)
    recent_attendance = Attendance.query.filter_by(student_id=student_id).filter(
        Attendance.timestamp > one_minute_ago
    ).first()
    
    if not recent_attendance:
        new_attendance = Attendance(student_id=student_id)
        db.session.add(new_attendance)
        db.session.commit()
        student = Student.query.get(student_id)
        print(f"Attendance logged for {student.name if student else 'Unknown'}")

@app.route('/video')
@login_required
def video():
    return render_template('video.html')

if __name__ == '__main__':
    app.run(debug=True)
