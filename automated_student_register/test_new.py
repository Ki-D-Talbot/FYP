from flask import Flask, render_template, Response, jsonify, request, redirect, url_for
from flask_sqlalchemy import SQLAlchemy
from flask_login import LoginManager, UserMixin, login_user, login_required, logout_user, current_user
import subprocess
import cv2
import numpy as np
import os
import time
from datetime import datetime, timedelta
import threading

app = Flask(__name__)
app.secret_key = 'your_secret_key'
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///database.db'
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False
db = SQLAlchemy(app)

# Initialize Flask-Login
login_manager = LoginManager()
login_manager.init_app(app)
login_manager.login_view = 'login'

# Database models
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

# Basic routes
@app.route('/')
@login_required
def index():
    return render_template('admin.html')

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

@app.route('/create_admin')
def create_admin():
    if User.query.filter_by(username='admin').first():
        return "Admin user already exists"
    admin = User(username='admin', password='admin123')
    db.session.add(admin)
    db.session.commit()
    return "Admin created successfully"

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
        
        new_student = Student(name=name, email=email, class_name=class_name, photo_path=photo_path)
        db.session.add(new_student)
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

# Camera handling routes - STATIC PHOTO APPROACH

@app.route('/video')
@login_required
def video():
    return render_template('photo_capture.html')

@app.route('/capture_photo')
@login_required
def capture_photo():
    """Capture a single photo using libcamera-jpeg for reliable operation"""
    try:
        # Kill any existing camera processes first
        subprocess.run(["pkill", "-f", "libcamera"], stderr=subprocess.DEVNULL)
        time.sleep(0.5)  # Give time for cleanup
        
        # Create directory for photos if needed
        os.makedirs("static/captures", exist_ok=True)
        
        # Generate unique filename with timestamp
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_path = f"static/captures/photo_{timestamp}.jpg"
        
        # Use libcamera-jpeg to take a photo
        process = subprocess.run(
            ["libcamera-jpeg", "-o", output_path, "--immediate"],
            timeout=3,
            capture_output=True,
            text=True
        )
        
        if not os.path.exists(output_path):
            return jsonify({
                "success": False,
                "message": f"Failed to capture image: {process.stderr}"
            })
        
        # Process the image for face detection
        img = cv2.imread(output_path)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
        faces = face_cascade.detectMultiScale(gray, 1.1, 4)
        
        # Draw rectangles around faces and log attendance
        detected_students = []
        for (x, y, w, h) in faces:
            # Draw rectangle
            cv2.rectangle(img, (x, y), (x+w, y+h), (0, 255, 0), 2)
            
            # For demo purposes, log attendance for first student
            students = Student.query.all()
            if students:
                student = students[0]
                log_attendance(student.student_id)
                detected_students.append(student.name)
                cv2.putText(img, student.name, (x, y-10), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
        
        # Save the processed image
        processed_path = f"static/captures/processed_{timestamp}.jpg"
        cv2.imwrite(processed_path, img)
        
        return jsonify({
            "success": True,
            "original_image": output_path,
            "processed_image": processed_path,
            "faces_detected": len(faces),
            "students_recognized": detected_students
        })
    except Exception as e:
        return jsonify({
            "success": False,
            "message": f"Error: {str(e)}"
        })

def log_attendance(student_id):
    """Log attendance for a student"""
    try:
        # Prevent duplicate logs within short timeframe
        one_minute_ago = datetime.now() - timedelta(minutes=1)
        recent_attendance = Attendance.query.filter_by(student_id=student_id).filter(
            Attendance.timestamp > one_minute_ago
        ).first()
        
        if not recent_attendance:
            new_attendance = Attendance(student_id=student_id)
            db.session.add(new_attendance)
            db.session.commit()
            print(f"Attendance logged for student ID: {student_id}")
    except Exception as e:
        print(f"Error logging attendance: {e}")

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

if __name__ == '__main__':
    # Ensure no camera processes are running when we start
    subprocess.run(["pkill", "-f", "libcamera"], stderr=subprocess.DEVNULL)
    time.sleep(1)
    
    # Run the app
    app.run(host='0.0.0.0', port=5000, debug=False)
