#!/usr/bin/env python3
"""
Camera service for automated student attendance monitoring.
Uses picamera2 directly for more reliable face detection and recognition.
"""
import os
import time
import sqlite3
import signal
import sys
import cv2
import numpy as np
from datetime import datetime
from picamera2 import Picamera2, Preview

# Flag to control the main loop
running = True

# Create necessary directories
os.makedirs('static/current', exist_ok=True)
os.makedirs('student_faces', exist_ok=True)

def signal_handler(sig, frame):
    global running
    print("Stopping camera service...")
    running = False
    sys.exit(0)

# Register signal handlers for graceful shutdown
signal.signal(signal.SIGINT, signal_handler)
signal.signal(signal.SIGTERM, signal_handler)

def log_message(message):
    """Log a message with timestamp"""
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print(f"[{timestamp}] {message}")
    
    # Also log to a file
    with open("camera_service.log", "a") as f:
        f.write(f"[{timestamp}] {message}\n")

def log_attendance(student_id, db_path='database.db'):
    """Log attendance in the database"""
    try:
        # Connect to the database
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()
        
        # Check for recent attendance to prevent duplicates
        cursor.execute(
            "SELECT * FROM attendance WHERE student_id = ? AND timestamp > datetime('now', '-1 minute')",
            (student_id,)
        )
        recent = cursor.fetchone()
        
        if not recent:
            # Log new attendance
            cursor.execute(
                "INSERT INTO attendance (student_id, timestamp) VALUES (?, datetime('now'))",
                (student_id,)
            )
            conn.commit()
            
            # Get student name for logging
            cursor.execute("SELECT name FROM student WHERE student_id = ?", (student_id,))
            student_name = cursor.fetchone()
            if student_name:
                log_message(f"Logged attendance for {student_name[0]} (ID: {student_id})")
            else:
                log_message(f"Logged attendance for student ID: {student_id}")
        
        conn.close()
        return True
    except Exception as e:
        log_message(f"Database error: {e}")
        return False

def recognize_student(face_img, conn):
    """
    Recognize a student face by comparing with stored faces
    
    Args:
        face_img: The detected face image
        conn: Database connection
        
    Returns:
        student_id, name if recognized, or None, None if not recognized
    """
    try:
        # Convert to proper size for recognition
        face_img_resized = cv2.resize(face_img, (160, 160))
        
        # Get all student IDs and their photo paths from the database
        cursor = conn.cursor()
        cursor.execute("SELECT student_id, name, photo_path FROM student WHERE photo_path IS NOT NULL")
        students = cursor.fetchall()
        
        if not students:
            log_message("No students with registered faces found in database")
            return None, None
        
        best_match_id = None
        best_match_name = None
        best_match_score = 0
        threshold = 0.65  # Minimum similarity threshold
        
        for student_id, name, photo_path in students:
            # Check if the photo path exists
            if not os.path.exists(photo_path):
                log_message(f"Student photo not found: {photo_path}")
                continue
            
            # Load the stored face image
            stored_face = cv2.imread(photo_path, cv2.IMREAD_GRAYSCALE)
            if stored_face is None:
                log_message(f"Failed to load student photo: {photo_path}")
                continue
            
            # Resize stored face to match
            stored_face_resized = cv2.resize(stored_face, (160, 160))
            
            # Compare faces using template matching
            # This is a simple comparison - in production you'd use a proper face recognition model
            result = cv2.matchTemplate(face_img_resized, stored_face_resized, cv2.TM_CCOEFF_NORMED)
            similarity = np.max(result)
            
            log_message(f"Similarity with {name} (ID: {student_id}): {similarity:.2f}")
            
            if similarity > threshold and similarity > best_match_score:
                best_match_score = similarity
                best_match_id = student_id
                best_match_name = name
        
        if best_match_id is not None:
            log_message(f"Recognized student: {best_match_name} (ID: {best_match_id}) with confidence: {best_match_score:.2f}")
            return best_match_id, best_match_name
        else:
            log_message(f"No matching student found above threshold ({threshold})")
            return None, None
            
    except Exception as e:
        log_message(f"Error in face recognition: {e}")
        return None, None

# Replace the face detection and recognition part in process_frame function
def process_frame(frame_path):
    """Process a frame for face detection and recognition"""
    try:
        if not os.path.exists(frame_path):
            log_message(f"Error: Frame file not found: {frame_path}")
            return False
        
        # Load the image
        img = cv2.imread(frame_path)
        if img is None:
            log_message(f"Error: Could not read image: {frame_path}")
            return False
        
        # Save original image for debugging
        debug_original_path = 'static/current/original.jpg'
        cv2.imwrite(debug_original_path, img)
        
        # Convert to grayscale for face detection
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        
        # Attempt to load the face cascade
        face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
        if face_cascade.empty():
            log_message("Error: Face cascade is empty - checking alternate paths")
            # Try alternate installation paths
            alternate_paths = [
                '/usr/local/share/opencv4/haarcascades/haarcascade_frontalface_default.xml',
                '/usr/share/opencv4/haarcascades/haarcascade_frontalface_default.xml'
            ]
            for path in alternate_paths:
                if os.path.exists(path):
                    log_message(f"Trying alternate cascade path: {path}")
                    face_cascade = cv2.CascadeClassifier(path)
                    if not face_cascade.empty():
                        log_message("Successfully loaded cascade from alternate path")
                        break
        
        # Try with more permissive parameters
        faces = face_cascade.detectMultiScale(
            gray, 
            scaleFactor=1.1, 
            minNeighbors=4,
            minSize=(30, 30)
        )
        
        if len(faces) == 0:
            # Try again with more permissive parameters
            faces = face_cascade.detectMultiScale(
                gray, 
                scaleFactor=1.2, 
                minNeighbors=3,
                minSize=(30, 30)
            )
        
        if len(faces) == 0:
            log_message("No faces detected")
            cv2.putText(img, "No faces detected", (50, 50), 
                      cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
            cv2.imwrite('static/current/frame.jpg', img)
            return False
        
        # Connect to the database for student recognition
        conn = sqlite3.connect('database.db')
        
        # Process detected faces
        students_detected = []
        for i, (x, y, w, h) in enumerate(faces):
            # Draw rectangle around face
            cv2.rectangle(img, (x, y), (x+w, y+h), (0, 255, 0), 2)
            
            # Extract face region for recognition
            face_img = gray[y:y+h, x:x+w]
            
            # Save detected face for debugging
            face_path = f'static/current/face_{i}.jpg'
            cv2.imwrite(face_path, face_img)
            
            # Try to recognize the student
            student_id, name = recognize_student(face_img, conn)
            
            if student_id:
                # Log attendance for recognized student
                log_attendance(student_id)
                students_detected.append(name)
                
                # Add name label above face with green color (recognized)
                label = f"Student: {name}"
                cv2.putText(img, label, (x, y-10), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            else:
                # Add unknown label with red color
                cv2.putText(img, "Unknown", (x, y-10), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        
        conn.close()
        
        # Add timestamp
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        cv2.putText(img, timestamp, (10, img.shape[0] - 10), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
        
        # Save the processed frame for the web app to use
        processed_frame_path = 'static/current/frame.jpg'
        cv2.imwrite(processed_frame_path, img)
        
        # Save detection results for the web app
        if students_detected:
            detection_text = f"Detected students: {', '.join(students_detected)}"
            log_message(detection_text)
            
            # Write detection info to a file that the web app can read
            with open('static/current/detection_results.txt', 'w') as f:
                f.write(f"Timestamp: {timestamp}\n")
                f.write(detection_text)
        
        return True
    except Exception as e:
        log_message(f"Error processing frame: {e}")
        return False

def recognize_face(face_img, conn):
    """
    Simple face recognition based on histogram comparison.
    
    In a real implementation, you would use a more sophisticated 
    face recognition algorithm like face_recognition library or a deep learning model.
    """
    try:
        cursor = conn.cursor()
        cursor.execute("SELECT student_id, name FROM student")
        students = cursor.fetchall()
        
        best_match = None
        best_score = 0
        
        # Resize for consistency
        face_img_resized = cv2.resize(face_img, (100, 100))
        face_hist = cv2.calcHist([face_img_resized], [0], None, [256], [0, 256])
        cv2.normalize(face_hist, face_hist, 0, 1, cv2.NORM_MINMAX)
        
        for student_id, name in students:
            # Check if student has a face image
            student_face_path = f'student_faces/{student_id}/face.jpg'
            if os.path.exists(student_face_path):
                student_face = cv2.imread(student_face_path, cv2.IMREAD_GRAYSCALE)
                if student_face is not None:
                    student_face_resized = cv2.resize(student_face, (100, 100))
                    student_hist = cv2.calcHist([student_face_resized], [0], None, [256], [0, 256])
                    cv2.normalize(student_hist, student_hist, 0, 1, cv2.NORM_MINMAX)
                    
                    # Compare histograms
                    score = cv2.compareHist(face_hist, student_hist, cv2.HISTCMP_CORREL)
                    
                    if score > best_score and score > 0.5:  # Threshold of 0.5 for similarity
                        best_score = score
                        best_match = (student_id, name, score)
        
        return best_match
    except Exception as e:
        log_message(f"Error in face recognition: {e}")
        return None

def main():
    """Main function to run the camera service"""
    log_message("Starting camera monitoring service")
    
    try:
        # Initialize Picamera2
        picam2 = Picamera2()
        config = picam2.create_preview_configuration(main={"size": (1280, 720)})
        picam2.configure(config)
        picam2.start()
        
        # Allow camera to warm up
        time.sleep(2)
        
        # Load face detector
        try:
            face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
            if face_cascade.empty():
                log_message("Error: Face cascade is empty - checking alternate paths")
                # Try alternate installation paths
                alternate_paths = [
                    '/usr/local/share/opencv4/haarcascades/haarcascade_frontalface_default.xml',
                    '/usr/share/opencv4/haarcascades/haarcascade_frontalface_default.xml'
                ]
                for path in alternate_paths:
                    if os.path.exists(path):
                        log_message(f"Trying alternate cascade path: {path}")
                        face_cascade = cv2.CascadeClassifier(path)
                        if not face_cascade.empty():
                            log_message("Successfully loaded cascade from alternate path")
                            break
                
                if face_cascade.empty():
                    log_message("Fatal error: Could not load face cascade classifier")
                    return
        except Exception as e:
            log_message(f"Error loading face cascade: {e}")
            return
        
        # Create a connection to the database
        conn = sqlite3.connect('database.db', check_same_thread=False)
        
        # Create a blank initial frame
        blank_frame = np.zeros((720, 1280, 3), dtype=np.uint8)
        cv2.putText(blank_frame, "Camera starting...", (50, 360), 
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        cv2.imwrite('static/current/frame.jpg', blank_frame)
        
        # Main loop
        fps_limit = 5  # limit to 5 frames per second
        frame_interval = 1.0 / fps_limit
        detection_interval = 0.5  # Run face detection every 0.5 seconds
        last_detection_time = 0
        
        while running:
            try:
                loop_start = time.time()
                
                # Capture frame
                frame = picam2.capture_array()
                
                # Convert to BGR (OpenCV format)
                if len(frame.shape) == 3 and frame.shape[2] == 3:  # If RGB
                    frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
                
                # Make a copy for processing
                display_frame = frame.copy()
                
                # Convert to grayscale for face detection
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                
                # Run face detection at specified interval
                current_time = time.time()
                if current_time - last_detection_time >= detection_interval:
                    last_detection_time = current_time
                    
                    # Detect faces with multiple parameter combinations for better accuracy
                    faces = []
                    for scale in [1.1, 1.2]:
                        for min_neighbors in [4, 5]:
                            detected_faces = face_cascade.detectMultiScale(
                                gray, 
                                scaleFactor=scale, 
                                minNeighbors=min_neighbors,
                                minSize=(30, 30)
                            )
                            
                            if len(detected_faces) > 0:
                                faces = detected_faces
                                break
                        
                        if len(faces) > 0:
                            break
                    
                    # Process detected faces
                    if len(faces) > 0:
                        log_message(f"Detected {len(faces)} faces")
                        
                        for i, (x, y, w, h) in enumerate(faces):
                            # Draw rectangle around face
                            cv2.rectangle(display_frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
                            
                            # Extract face for recognition
                            face_img = gray[y:y+h, x:x+w]
                            
                            # Save detected face for debugging
                            face_path = f'static/current/face_{i}.jpg'
                            cv2.imwrite(face_path, face_img)
                            
                            # Recognize the face
                            match = recognize_face(face_img, conn)
                            
                            if match:
                                student_id, name, score = match
                                # Log attendance
                                log_attendance(student_id)
                                
                                # Add name label above face
                                label = f"{name} ({score:.2f})"
                                cv2.putText(display_frame, label, (x, y-10), 
                                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                            else:
                                # Unknown person
                                cv2.putText(display_frame, "Unknown", (x, y-10), 
                                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                    else:
                        cv2.putText(display_frame, "No faces detected", (50, 50), 
                                  cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
                
                # Add timestamp
                timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                cv2.putText(display_frame, timestamp, (10, display_frame.shape[0] - 10), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
                
                # Save the processed frame for the web app to use
                cv2.imwrite('static/current/frame.jpg', display_frame)
                
                # Calculate remaining time to maintain frame rate
                processing_time = time.time() - loop_start
                sleep_time = max(0, frame_interval - processing_time)
                time.sleep(sleep_time)
                
            except Exception as e:
                log_message(f"Error in main loop: {e}")
                time.sleep(1)  # Wait before continuing
        
        # Cleanup
        picam2.stop()
        conn.close()
        log_message("Camera service stopped")
        
    except Exception as e:
        log_message(f"Fatal error: {e}")

if __name__ == "__main__":
    # Make sure we're the only instance running
    try:
        import psutil
        current_pid = os.getpid()
        for proc in psutil.process_iter(['pid', 'name', 'cmdline']):
            if proc.info['pid'] != current_pid and 'camera_service.py' in ' '.join(proc.info['cmdline']):
                log_message(f"Another instance is already running (PID: {proc.info['pid']}). Exiting.")
                sys.exit(1)
    except ImportError:
        # psutil not available, skip this check
        pass
    
    main()