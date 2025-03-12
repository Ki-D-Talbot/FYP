#!/usr/bin/env python3
"""
Simple camera service for automated student attendance monitoring.
Uses libcamera-jpeg for reliable operation.
"""
import os
import time
import subprocess
import sqlite3
from datetime import datetime
import cv2
import signal
import sys
import numpy as np

# Flag to control the main loop
running = True

def signal_handler(sig, frame):
    global running
    print("Stopping camera service...")
    running = False
    sys.exit(0)

# Register signal handlers for graceful shutdown
signal.signal(signal.SIGINT, signal_handler)
signal.signal(signal.SIGTERM, signal_handler)

# Create necessary directories
os.makedirs('static/current', exist_ok=True)
os.makedirs('camera_frames', exist_ok=True)

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
            log_message(f"Logged attendance for student ID: {student_id}")
        
        conn.close()
        return True
    except Exception as e:
        log_message(f"Database error: {e}")
        return False

def capture_frame():
    """Capture a frame using libcamera-jpeg subprocess"""
    try:
        # Make sure no existing process is using the camera
        try:
            subprocess.run(["pkill", "-f", "libcamera"], stderr=subprocess.DEVNULL)
            time.sleep(0.5)  # Give time for cleanup
        except:
            pass
        
        # Generate unique filename
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_path = f"camera_frames/frame_{timestamp}.jpg"
        
        # Capture a frame with more detailed error handling
        try:
            result = subprocess.run(
                ["libcamera-jpeg", "-o", output_path, "--immediate", "--width", "1280", "--height", "720"],
                capture_output=True,
                timeout=3
            )
            
            if result.returncode != 0:
                log_message(f"libcamera-jpeg returned error code {result.returncode}: {result.stderr.decode()}")
                return None
                
        except subprocess.TimeoutExpired:
            log_message("Timeout while capturing frame")
            return None
        
        if os.path.exists(output_path):
            log_message(f"Captured frame: {output_path}")
            return output_path
        else:
            log_message(f"Failed to capture frame - file not created")
            return None
    except Exception as e:
        log_message(f"Error capturing frame: {e}")
        return None

def process_frame(frame_path):
    """Process a frame for face detection with enhanced debugging"""
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
        log_message(f"Saved original frame to {debug_original_path}")
        
        # Convert to grayscale for face detection
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        
        # Save grayscale image for debugging
        debug_gray_path = 'static/current/gray.jpg'
        cv2.imwrite(debug_gray_path, gray)
        
        # Attempt to load the face cascade
        try:
            face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
            if face_cascade.empty():
                log_message("Error: Face cascade is empty - check OpenCV installation")
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
        except Exception as e:
            log_message(f"Error loading face cascade: {e}")
            return False
        
        # Try with more permissive parameters
        faces = []
        detection_success = False
        
        # Try different parameters for improved detection rate
        scale_factors = [1.05, 1.1, 1.2]
        min_neighbors_options = [3, 4, 5]
        
        for scale in scale_factors:
            for min_neighbors in min_neighbors_options:
                try:
                    # Try to detect faces with current parameters
                    current_faces = face_cascade.detectMultiScale(
                        gray, 
                        scaleFactor=scale, 
                        minNeighbors=min_neighbors,
                        minSize=(30, 30)  # Minimum face size
                    )
                    
                    if len(current_faces) > 0:
                        faces = current_faces
                        log_message(f"Detected {len(faces)} faces with scale={scale}, minNeighbors={min_neighbors}")
                        detection_success = True
                        break
                except Exception as e:
                    log_message(f"Error during face detection with scale={scale}, minNeighbors={min_neighbors}: {e}")
            
            if detection_success:
                break
        
        if not detection_success:
            log_message("No faces detected with any parameter combination")
            
            # Save a blank processed frame to avoid errors
            cv2.putText(img, "No faces detected", (50, 50), 
                      cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
            cv2.imwrite('static/current/frame.jpg', img)
            
            return False
        
        # Process detected faces
        students_detected = []
        for i, (x, y, w, h) in enumerate(faces):
            # Draw rectangle around face
            cv2.rectangle(img, (x, y), (x+w, y+h), (0, 255, 0), 2)
            
            # Try to identify the student
            face_img = gray[y:y+h, x:x+w]
            
            # Save detected face for debugging
            face_path = f'static/current/face_{i}.jpg'
            cv2.imwrite(face_path, face_img)
            
            # For now, just log attendance for student ID 1
            # In a real implementation, you'd compare with student photos
            try:
                conn = sqlite3.connect('database.db')
                cursor = conn.cursor()
                
                # Get first student ID (simplified for this example)
                cursor.execute("SELECT student_id, name FROM student LIMIT 1")
                result = cursor.fetchone()
                
                if result:
                    student_id, name = result
                    log_attendance(student_id)
                    students_detected.append(name)
                    
                    # Add name label above face
                    label = f"Student: {name}"
                    cv2.putText(img, label, (x, y-10), 
                                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                
                conn.close()
            except Exception as e:
                log_message(f"Error getting student info: {e}")
        
        # Add timestamp
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        cv2.putText(img, timestamp, (10, img.shape[0] - 10), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
        
        # Save the processed frame for the web app to use
        processed_frame_path = 'static/current/frame.jpg'
        cv2.imwrite(processed_frame_path, img)
        log_message(f"Saved processed frame to {processed_frame_path}")
        
        # Also save detection results for the web app
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

def main():
    """Main function to run the camera service"""
    log_message("Starting camera monitoring service")
    
    try:
        # Check if we can access the face cascade file
        cascade_path = cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
        if not os.path.exists(cascade_path):
            log_message(f"Warning: Face cascade file not found at {cascade_path}")
            # Try alternate paths
            alternate_paths = [
                '/usr/local/share/opencv4/haarcascades/haarcascade_frontalface_default.xml',
                '/usr/share/opencv4/haarcascades/haarcascade_frontalface_default.xml'
            ]
            cascade_found = False
            for path in alternate_paths:
                if os.path.exists(path):
                    log_message(f"Found face cascade file at alternate path: {path}")
                    cascade_found = True
                    break
            
            if not cascade_found:
                log_message("Error: Could not find face cascade file in any standard location")
        else:
            log_message(f"Found face cascade file at {cascade_path}")
        
        # Create a blank initial frame to avoid errors
        blank_frame = np.zeros((480, 640, 3), dtype=np.uint8)
        cv2.putText(blank_frame, "Camera starting...", (50, 240), 
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        cv2.imwrite('static/current/frame.jpg', blank_frame)
        
        # Main loop
        capture_failures = 0
        while running:
            try:
                # Capture a frame
                frame_path = capture_frame()
                
                if frame_path:
                    # Process the frame
                    process_frame(frame_path)
                    capture_failures = 0  # Reset failure counter on success
                else:
                    capture_failures += 1
                    log_message(f"Frame capture failed ({capture_failures} consecutive failures)")
                    
                    if capture_failures >= 5:
                        log_message("Multiple consecutive failures, attempting to restart camera...")
                        # Try to reset camera
                        try:
                            subprocess.run(["pkill", "-f", "libcamera"], stderr=subprocess.DEVNULL)
                            time.sleep(1)  # Give time for cleanup
                            capture_failures = 0  # Reset counter
                        except Exception as e:
                            log_message(f"Error resetting camera: {e}")
                
                # Sleep between captures
                time.sleep(2)  # Capture every 2 seconds
            except Exception as e:
                log_message(f"Error in main loop: {e}")
                time.sleep(5)  # Longer delay on error
        
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