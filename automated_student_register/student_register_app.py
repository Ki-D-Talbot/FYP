from flask import Flask, render_template, Response, request, redirect, url_for, jsonify
from flask_sqlalchemy import SQLAlchemy
from flask_login import LoginManager, UserMixin, login_user, logout_user, login_required, current_user
import os
import time
import subprocess
import cv2
import numpy as np
from datetime import datetime

app = Flask(__name__)
app.secret_key = 'your_secret_key'
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///database.db'
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False
db = SQLAlchemy(app)

# Initialize Flask-Login
login_manager = LoginManager()
login_manager.init_app(app)
login_manager.login_view = 'login'

# Database models (same as before)
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

# Create database tables
with app.app_context():
    db.create_all()

@login_manager.user_loader
def load_user(user_id):
    return db.session.get(User, int(user_id))

# Routes
@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        username = request.form['username']
        password = request.form['password']
        user = User.query.filter_by(username=username).first()
        if user and user.password == password:
            login_user(user)
            return redirect(url_for('index'))
        else:
            return render_template('login.html', error='Invalid credentials')
    return render_template('login.html')

@app.route('/logout')
@login_required
def logout():
    logout_user()
    return redirect(url_for('login'))

@app.route('/')
@login_required
def index():
    return render_template('admin.html')

@app.route('/students')
@login_required
def list_students():
    students = Student.query.all()
    return render_template('students.html', students=students)

@app.route('/add_student', methods=['GET', 'POST'])
@login_required
def add_student():
    if request.method == 'POST':
        name = request.form['name']
        email = request.form.get('email')
        class_name = request.form.get('class_name')
        photo_path = request.form.get('photo_path')
        
        student = Student(name=name, email=email, class_name=class_name, photo_path=photo_path)
        db.session.add(student)
        db.session.commit()
        
        return redirect(url_for('list_students'))
    return render_template('add_student.html')

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

@app.route('/create_admin')
def create_admin():
    if User.query.filter_by(username='admin').first():
        return "Admin user already exists"
    
    admin = User(username='admin', password='admin123')
    db.session.add(admin)
    db.session.commit()
    return "Admin created successfully"

# Camera monitoring interface
@app.route('/video')
@login_required
def video():
    # Check if camera service is running
    camera_running = is_camera_service_running()
    return render_template('video.html', camera_running=camera_running)

def is_camera_service_running():
    """Check if the camera service is running"""
    try:
        output = subprocess.check_output(["pgrep", "-f", "camera_service.py"]).decode().strip()
        return bool(output)
    except:
        return False

@app.route('/video_feed')
def video_feed():
    """Stream the current frame from the background process"""
    def generate_frames():
        while True:
            try:
                # Check if current frame exists
                if os.path.exists('static/current/frame.jpg'):
                    # Get modification time
                    mod_time = os.path.getmtime('static/current/frame.jpg')
                    current_time = time.time()
                    
                    # Only use frame if it's recent (less than 5 seconds old)
                    if current_time - mod_time < 5:
                        frame = cv2.imread('static/current/frame.jpg')
                        ret, buffer = cv2.imencode('.jpg', frame)
                        frame_bytes = buffer.tobytes()
                        
                        yield (b'--frame\r\n'
                               b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')
                    else:
                        # Frame is too old, show placeholder
                        frame = create_placeholder_frame("Camera feed not updating")
                        ret, buffer = cv2.imencode('.jpg', frame)
                        frame_bytes = buffer.tobytes()
                        
                        yield (b'--frame\r\n'
                               b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')
                else:
                    # No frame available, show placeholder
                    frame = create_placeholder_frame("Camera not running")
                    ret, buffer = cv2.imencode('.jpg', frame)
                    frame_bytes = buffer.tobytes()
                    
                    yield (b'--frame\r\n'
                           b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')
            except Exception as e:
                print(f"Error in generate_frames: {e}")
                
                # On error, show placeholder
                frame = create_placeholder_frame(f"Error: {str(e)}")
                ret, buffer = cv2.imencode('.jpg', frame)
                frame_bytes = buffer.tobytes()
                
                yield (b'--frame\r\n'
                       b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')
            
            # Wait before next frame
            time.sleep(0.2)
    
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

def create_placeholder_frame(message):
    """Create a placeholder frame with an error message"""
    frame = np.zeros((480, 640, 3), dtype=np.uint8)
    cv2.putText(frame, message, (50, 240), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
    return frame

@app.route('/start_camera_service')
@login_required
def start_camera_service():
    """Start the camera service if it's not running"""
    if not is_camera_service_running():
        try:
            # Create necessary directories
            os.makedirs('static/current', exist_ok=True)
            
            # Use the simplified camera service
            script_path = "camera_service.py"
            
            # Make sure the script is executable
            if os.path.exists(script_path):
                os.chmod(script_path, 0o755)
            else:
                return jsonify({
                    "success": False, 
                    "message": f"Error: {script_path} file not found"
                })
            
            # Start the camera service with output logging
            with open("camera_start.log", "w") as log_file:
                process = subprocess.Popen(
                    ["python3", script_path],
                    stdout=log_file,
                    stderr=log_file
                )
            
            # Give it a moment to start
            time.sleep(2)
            
            # Check if it's running
            if is_camera_service_running():
                return jsonify({
                    "success": True, 
                    "message": "Camera service started successfully"
                })
            else:
                return jsonify({
                    "success": False, 
                    "message": "Camera service failed to start. Check camera_start.log for details."
                })
                
        except Exception as e:
            return jsonify({
                "success": False, 
                "message": f"Error starting camera service: {str(e)}"
            })
    else:
        return jsonify({
            "success": True, 
            "message": "Camera service already running"
        })

@app.route('/stop_camera_service')
@login_required
def stop_camera_service():
    """Stop the camera service if it's running"""
    if is_camera_service_running():
        try:
            # Kill the camera service process
            subprocess.run(["pkill", "-f", "camera_service.py"])
            return jsonify({"success": True, "message": "Camera service stopped"})
        except Exception as e:
            return jsonify({"success": False, "message": f"Error stopping camera service: {str(e)}"})
    else:
        return jsonify({"success": True, "message": "Camera service not running"})

@app.route('/check_attendance')
@login_required
def check_attendance():
    # Get the most recent 10 attendance records
    recent_attendance = db.session.query(
        Attendance, Student.name
    ).join(
        Student, Attendance.student_id == Student.student_id
    ).order_by(
        Attendance.timestamp.desc()
    ).limit(10).all()
    
    attendance_data = []
    for attendance, name in recent_attendance:
        attendance_data.append({
            'id': attendance.attendance_id,
            'student_name': name,
            'timestamp': attendance.timestamp.strftime('%Y-%m-%d %H:%M:%S')
        })
    
    return render_template('check_attendance.html', attendance_data=attendance_data)

@app.route('/capture_face/<int:student_id>')
@login_required
def capture_face(student_id):
    """Capture a photo for student face recognition"""
    student = Student.query.get_or_404(student_id)
    return render_template('capture_face.html', student=student)

@app.route('/save_face', methods=['POST'])
@login_required
def save_face():
    """Save a captured face photo for a student"""
    if 'image' not in request.files:
        return jsonify({"error": "No image provided"}), 400
    
    image = request.files['image']
    student_id = request.form.get('student_id')
    
    if not student_id:
        return jsonify({"error": "No student ID provided"}), 400
    
    # Create directory for student faces
    os.makedirs('student_faces', exist_ok=True)
    student_folder = os.path.join('student_faces', student_id)
    os.makedirs(student_folder, exist_ok=True)
    
    # Save the image
    image_path = os.path.join(student_folder, 'face.jpg')
    try:
        image.save(image_path)
        
        # Update student record
        student = Student.query.get(student_id)
        if student:
            student.photo_path = image_path
            db.session.commit()
        
        return jsonify({"success": True, "message": "Face saved successfully"}), 200
    except Exception as e:
        print(f"Error saving face: {e}")
        return jsonify({"error": str(e)}), 500


if __name__ == '__main__':
    # Create necessary directories
    os.makedirs('static/current', exist_ok=True)
    
    # Run the application
    app.run(host='0.0.0.0', port=5000, debug=True)
