from flask import Flask, render_template, request, redirect, url_for, flash, session, Response, jsonify
from flask_sqlalchemy import SQLAlchemy
from flask_login import LoginManager, UserMixin, login_user, login_required, logout_user, current_user
import cv2
import datetime
import os
import numpy as np
import subprocess
import threading
import time
import traceback
import signal
import sys
import tflite_runtime.interpreter as tflite
from sklearn.metrics.pairwise import cosine_similarity

camera_service_process = None
camera_lock = threading.Lock()

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

# Add this near the top of your app where other directories are created
os.makedirs('static/current', exist_ok=True)

# Use app.app_context() to create tables
with app.app_context():
    db.create_all()  # Create tables

# Load facial recognition model
def load_model():
    try:
        model_path = 'facial_recognition_model.tflite'  
        interpreter = tflite.Interpreter(model_path=model_path)
        interpreter.allocate_tensors()
        print("Model loaded successfully")
        return interpreter
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
    
    # Use a consistent BASE_DIR for all file operations
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    student_folder = os.path.join(BASE_DIR, 'student_faces', student_id)
    os.makedirs(student_folder, exist_ok=True)
    
    image_path = os.path.join(student_folder, 'face.jpg')
    try:
        # Save the raw image
        image.save(image_path)
        
        # Process with face detection
        img = cv2.imread(image_path)
        if img is None:
            return jsonify({"error": "Could not read saved image"}), 500
            
        # Convert to grayscale for face detection
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        
        # Load face detector
        face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
        if face_cascade.empty():
            # Try alternate paths
            alternate_paths = [
                '/usr/local/share/opencv4/haarcascades/haarcascade_frontalface_default.xml',
                '/usr/share/opencv4/haarcascades/haarcascade_frontalface_default.xml'
            ]
            for path in alternate_paths:
                if os.path.exists(path):
                    face_cascade = cv2.CascadeClassifier(path)
                    if not face_cascade.empty():
                        break
                        
        if face_cascade.empty():
            return jsonify({"error": "Could not load face detection model"}), 500
                        
        # Detect faces with more permissive parameters
        faces = face_cascade.detectMultiScale(
            gray, 
            scaleFactor=1.1, 
            minNeighbors=3,  # Lower value to be more permissive
            minSize=(30, 30)
        )
        
        if len(faces) == 0:
            # Try again with even more permissive parameters
            faces = face_cascade.detectMultiScale(
                gray, 
                scaleFactor=1.3, 
                minNeighbors=2,
                minSize=(20, 20)
            )
            
        if len(faces) == 0:
            return jsonify({"error": "No face detected in the captured image. Please try again with better lighting."}), 400
            
        # If we found faces, crop to the largest one for better recognition later
        if len(faces) > 0:
            # Find the largest face
            largest_face = max(faces, key=lambda f: f[2] * f[3])
            x, y, w, h = largest_face
            
            # Add some margin
            margin = int(w * 0.2)  # 20% margin
            x = max(0, x - margin)
            y = max(0, y - margin)
            w = min(img.shape[1] - x, w + 2 * margin)
            h = min(img.shape[0] - y, h + 2 * margin)
            
            # Crop and save
            face_img = img[y:y+h, x:x+w]
            cv2.imwrite(image_path, face_img)
        
        # Store the RELATIVE path in the database
        rel_path = os.path.join('student_faces', student_id, 'face.jpg')
        
        student = db.session.get(Student, student_id)
        if student:
            student.photo_path = rel_path
            db.session.commit()
        
        print(f"Face saved to {image_path}")
        print(f"Database updated with path: {rel_path}")
        
        return jsonify({"success": True, "message": "Face saved successfully"}), 200
    except Exception as e:
        print(f"Error saving face: {e}")
        return jsonify({"error": str(e)}), 500

@app.route('/basic_video_feed')
def basic_video_feed():
    def generate_frames():
        try:
            from picamera2 import Picamera2
            
            picam2 = Picamera2()
            config = picam2.create_preview_configuration()
            picam2.configure(config)
            picam2.start()
            
            try:
                while True:
                    # Capture frame
                    frame = picam2.capture_array()
                    
                    # Convert BGR to RGB if needed (depends on your picamera2 configuration)
                    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    
                    # Convert frame to JPEG
                    ret, buffer = cv2.imencode('.jpg', frame)
                    frame_bytes = buffer.tobytes()
                    
                    yield (b'--frame\r\n'
                           b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')
            finally:
                picam2.stop()
                print("Camera stopped")
        except Exception as e:
            print(f"Error in camera feed: {e}")
    
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/start_camera_service')
@login_required
def start_camera_service():
    global camera_service_process
    
    try:
        with camera_lock:
            # Check if service is already running
            if camera_service_process and camera_service_process.poll() is None:
                return jsonify({"status": "already_running", "message": "Camera service is already running"})
            
            # Kill any existing camera processes
            try:
                subprocess.run(["pkill", "-f", "libcamera"], stderr=subprocess.DEVNULL)
                time.sleep(0.5)  # Give time for cleanup
            except:
                pass
            
            # Get the absolute path to the script
            # Use the directory of the current script file to find camera_service.py
            app_dir = os.path.dirname(os.path.abspath(__file__))
            script_path = os.path.join(app_dir, "camera_service.py")
            
            # Fallback paths if the primary location doesn't work
            fallback_paths = [
                os.path.join(os.getcwd(), "camera_service.py"),
                os.path.join(os.path.dirname(os.getcwd()), "automated_student_register", "camera_service.py")
            ]
            
            # Check if the primary file exists
            if not os.path.exists(script_path):
                # Try fallback paths
                for path in fallback_paths:
                    if os.path.exists(path):
                        script_path = path
                        break
                else:  # No path was found
                    return jsonify({
                        "status": "error", 
                        "message": f"Camera service file not found. Searched in: {script_path} and {fallback_paths}"
                    })
            
            # Start the camera service as a subprocess with full path
            camera_service_process = subprocess.Popen(
                [sys.executable, script_path],
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE
            )
            
            # Give it time to start
            time.sleep(1)
            
            if camera_service_process.poll() is None:
                return jsonify({
                    "status": "success", 
                    "message": f"Camera service started successfully with path: {script_path}"
                })
            else:
                stderr = camera_service_process.stderr.read().decode()
                return jsonify({
                    "status": "error", 
                    "message": f"Failed to start camera service: {stderr}",
                    "path_used": script_path
                })
    except Exception as e:
        return jsonify({
            "status": "error", 
            "message": f"Error: {str(e)}",
            "traceback": traceback.format_exc()
        })
    
@app.route('/stop_camera_service')
@login_required
def stop_camera_service():
    global camera_service_process
    
    try:
        with camera_lock:
            if camera_service_process and camera_service_process.poll() is None:
                camera_service_process.terminate()
                time.sleep(0.5)
                if camera_service_process.poll() is None:
                    camera_service_process.kill()
                return jsonify({"status": "success", "message": "Camera service stopped"})
            else:
                return jsonify({"status": "not_running", "message": "Camera service is not running"})
    except Exception as e:
        return jsonify({"status": "error", "message": f"Error: {str(e)}"})

@app.route('/camera_service_status')
@login_required
def camera_service_status():
    global camera_service_process
    
    try:
        with camera_lock:
            if camera_service_process and camera_service_process.poll() is None:
                return jsonify({"status": "running"})
            else:
                return jsonify({"status": "stopped"})
    except Exception as e:
        return jsonify({"status": "error", "message": f"Error: {str(e)}"})

@app.route('/video_feed')
def video_feed():
    def generate_frames():
        last_frame_time = 0
        no_frame_count = 0
        
        while True:
            try:
                current_time = time.time()
                # Only process every 100ms to reduce CPU usage
                if current_time - last_frame_time < 0.1:
                    time.sleep(0.05)
                    continue
                    
                last_frame_time = current_time
                current_frame_path = 'static/current/frame.jpg'
                
                if os.path.exists(current_frame_path):
                    # Get modification time
                    mod_time = os.path.getmtime(current_frame_path)
                    
                    # Only use frame if it's recent (less than 5 seconds old)
                    if current_time - mod_time < 5:
                        # Read the current frame saved by the camera service
                        frame = cv2.imread(current_frame_path)
                        no_frame_count = 0  # Reset counter
                        
                        if frame is not None:
                            # Resize to reduce bandwidth if frame is large
                            if frame.shape[0] > 480 or frame.shape[1] > 640:
                                frame = cv2.resize(frame, (640, 480))
                                
                            # Convert frame to JPEG with lower quality for better performance
                            encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), 70]
                            ret, buffer = cv2.imencode('.jpg', frame, encode_param)
                            frame_bytes = buffer.tobytes()
                            
                            yield (b'--frame\r\n'
                                b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')
                            continue
                
                # If we get here, either the file doesn't exist or frame is too old
                no_frame_count += 1
                
                # Only create a new blank frame every 10 iterations to reduce CPU usage
                if no_frame_count % 10 == 1:
                    frame = np.zeros((240, 320, 3), dtype=np.uint8)  # Smaller frame for performance
                    
                    if os.path.exists(current_frame_path):
                        cv2.putText(frame, "Camera feed not updating", (20, 120), 
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
                    else:
                        cv2.putText(frame, "Camera not running", (20, 120), 
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
                    
                    ret, buffer = cv2.imencode('.jpg', frame)
                    frame_bytes = buffer.tobytes()
                    
                    yield (b'--frame\r\n'
                        b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')
                    
                time.sleep(0.1)  # Reduced sleep time
                
            except Exception as e:
                print(f"Error in generate_frames: {e}")
                time.sleep(0.5)
    
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/capture_feed')
def capture_feed():
    """A lighter version of the video feed for capture face page"""
    def generate_frames():
        try:
            # First try with PiCamera2 for Raspberry Pi
            try:
                from picamera2 import Picamera2
                
                picam2 = Picamera2()
                config = picam2.create_preview_configuration(main={"size": (640, 480)})
                picam2.configure(config)
                picam2.start()
                
                try:
                    while True:
                        frame = picam2.capture_array()
                        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                        
                        # Use lower quality encoding for performance
                        encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), 70]
                        ret, buffer = cv2.imencode('.jpg', frame, encode_param)
                        frame_bytes = buffer.tobytes()
                        
                        yield (b'--frame\r\n'
                               b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')
                        time.sleep(0.1)  # Reduce framerate for better performance
                finally:
                    picam2.stop()
            
            # If PiCamera fails, try OpenCV with standard webcam
            except (ImportError, Exception) as e:
                print(f"PiCamera error: {e}, falling back to OpenCV")
                cap = cv2.VideoCapture(0)
                cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
                cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
                
                try:
                    while True:
                        success, frame = cap.read()
                        if not success:
                            break
                            
                        # Use lower quality encoding for performance
                        encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), 70]
                        ret, buffer = cv2.imencode('.jpg', frame, encode_param)
                        frame_bytes = buffer.tobytes()
                        
                        yield (b'--frame\r\n'
                               b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')
                        time.sleep(0.1)  # Reduce framerate for better performance
                finally:
                    cap.release()
                    
        except Exception as e:
            print(f"Error in capture_feed: {e}")
            # Return a blank frame with error message
            frame = np.zeros((240, 320, 3), dtype=np.uint8)
            cv2.putText(frame, f"Camera error: {str(e)[:20]}", (10, 120), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
            ret, buffer = cv2.imencode('.jpg', frame)
            frame_bytes = buffer.tobytes()
            
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')
    
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

@app.route('/get_detection_results')
def get_detection_results():
    """API endpoint to get the latest detection results"""
    try:
        detection_file = 'static/current/detection_results.txt'
        if os.path.exists(detection_file):
            # Get file modification time
            mod_time = os.path.getmtime(detection_file)
            timestamp = datetime.datetime.fromtimestamp(mod_time).strftime('%Y-%m-%d %H:%M:%S')
            
            # Check if file is recent (less than 30 seconds old)
            time_diff = time.time() - mod_time
            is_recent = time_diff < 30
            
            # Read the file contents
            with open(detection_file, 'r') as f:
                content = f.read()
                
            return jsonify({
                'status': 'success',
                'content': content,
                'timestamp': timestamp,
                'is_recent': is_recent
            })
        else:
            return jsonify({
                'status': 'no_data',
                'message': 'No detection results available'
            })
    except Exception as e:
        return jsonify({
            'status': 'error',
            'message': str(e)
        })
    
@app.route('/today_attendance')
@login_required
def today_attendance():
    """API endpoint to get today's attendance"""
    try:
        # Get today's date at midnight
        today_start = datetime.datetime.now().replace(hour=0, minute=0, second=0, microsecond=0)
        
        # Query for today's attendance with student names
        attendance_records = db.session.query(
            Attendance, Student.name
        ).join(
            Student, Attendance.student_id == Student.student_id
        ).filter(
            Attendance.timestamp >= today_start
        ).order_by(
            Attendance.timestamp.desc()
        ).all()
        
        # Format the results
        students = []
        for attendance, name in attendance_records:
            students.append({
                'id': attendance.student_id,
                'name': name,
                'time': attendance.timestamp.strftime('%H:%M:%S')
            })
        
        # Get unique students (first appearance only)
        unique_students = []
        seen_ids = set()
        for student in students:
            if student['id'] not in seen_ids:
                unique_students.append(student)
                seen_ids.add(student['id'])
        
        return jsonify({
            'status': 'success',
            'total_count': len(unique_students),
            'students': unique_students
        })
    except Exception as e:
        return jsonify({
            'status': 'error',
            'message': str(e)
        })

@app.route('/debug_camera')
@login_required
def debug_camera():
    """Debugging endpoint for camera configuration"""
    # Check for camera service
    service_running = False
    with camera_lock:
        if camera_service_process and camera_service_process.poll() is None:
            service_running = True
    
    # Check for camera static directory
    static_dir_exists = os.path.exists('static/current')
    frame_exists = os.path.exists('static/current/frame.jpg')
    
    # Check if picamera2 is available
    picamera_available = False
    try:
        from picamera2 import Picamera2
        picamera_available = True
    except ImportError:
        pass
    
    # Get available video devices
    video_devices = []
    if os.path.exists('/dev'):
        for device in os.listdir('/dev'):
            if device.startswith('video'):
                video_devices.append(f"/dev/{device}")
    
    debug_info = {
        'camera_service_running': service_running,
        'static_directory_exists': static_dir_exists,
        'frame_file_exists': frame_exists,
        'picamera_available': picamera_available,
        'video_devices': video_devices,
        'python_executable': sys.executable,
        'working_directory': os.getcwd(),
        'app_directory': os.path.dirname(os.path.abspath(__file__))
    }
    
    # Check if frame is recent
    if frame_exists:
        mod_time = os.path.getmtime('static/current/frame.jpg')
        current_time = time.time()
        debug_info['frame_age_seconds'] = current_time - mod_time
    
    return jsonify(debug_info)

if __name__ == '__main__':
    app.run(debug=True)
