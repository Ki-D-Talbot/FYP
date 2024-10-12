from flask import Flask, render_template, request, redirect, url_for, flash, session, Response
from flask_sqlalchemy import SQLAlchemy
from flask_login import LoginManager, UserMixin, login_user, login_required, logout_user, current_user
import cv2
import datetime

app = Flask(__name__)
app.secret_key = 'your_secret_key'  # Change this to a random secret key
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

# Use app.app_context() to create tables
with app.app_context():
    db.create_all()  # Create tables

@login_manager.user_loader
def load_user(user_id):
    return User.query.get(int(user_id))

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

        return redirect(url_for('list_students'))  # Redirect to the student list after adding

    return render_template('add_student.html')

@app.route('/students')
@login_required
def list_students():
    students = Student.query.all()  # Fetch all students from the database
    return render_template('students.html', students=students)

@app.route('/create_admin')
def create_admin():
    admin = User(username='admin', password='admin123')  # Change this to a secure password
    db.session.add(admin)
    db.session.commit()
    return "Admin user created!"

@app.route('/edit_student/<int:student_id>', methods=['GET', 'POST'])
@login_required
def edit_student(student_id):
    student = Student.query.get_or_404(student_id)  # Fetch the student by ID

    if request.method == 'POST':
        # Update student information
        student.name = request.form['name']
        student.email = request.form.get('email')
        student.class_name = request.form.get('class_name')
        student.photo_path = request.form.get('photo_path')  # Update photo path if needed

        db.session.commit()  # Save changes to the database
        return redirect(url_for('list_students'))  # Redirect to the student list

    return render_template('edit_student.html', student=student)  # Render the edit form

@app.route('/video_feed')
def video_feed():
    # Load the Haar Cascade for face detection
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    camera = cv2.VideoCapture(0)

    def generate_frames():
        while True:
            success, frame = camera.read()  # Read a frame from the webcam
            if not success:
                break
            else:
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)  # Convert to grayscale
                faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5)

                # If a face is detected, log the attendance
                if len(faces) > 0:
                    for (x, y, w, h) in faces:
                        cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)  # Draw rectangle around face
                    log_attendance()  # Call the function to log attendance

                # Convert the frame to JPEG format
                ret, buffer = cv2.imencode('.jpg', frame)
                frame = buffer.tobytes()
                yield (b'--frame\r\n'
                       b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

def log_attendance():
    # Log the attendance in the database
    # You can modify this function to log the specific student based on your logic
    new_attendance = Attendance(student_id=1)  # Replace with actual student ID logic
    db.session.add(new_attendance)
    db.session.commit()

@app.route('/video')
@login_required
def video():
    return render_template('video.html')

if __name__ == '__main__':
    app.run(debug=True)
