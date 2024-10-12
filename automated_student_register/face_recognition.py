import cv2
import os
import numpy as np
from datetime import datetime
from picamera2 import Picamera2 # Adjust this import based on your library
import face_recognition  # Make sure to install face_recognition library
from automated_student_register.app import db, Student, Attendance   # Adjust the import based on your Flask app structure
import time

# Initialize the camera
camera = Picamera2()

# Load known faces
known_face_encodings = []
known_face_names = []

# Load known faces from the directory
known_faces_dir = 'C:/Users/kital/Documents/GitHub/FYP/automated_student_register/static/student_images/'
for filename in os.listdir(known_faces_dir):
    if filename.endswith('.jpg') or filename.endswith('.png'):
        image_path = os.path.join(known_faces_dir, filename)
        image = face_recognition.load_image_file(image_path)
        encoding = face_recognition.face_encodings(image)[0]
        known_face_encodings.append(encoding)
        known_face_names.append(os.path.splitext(filename)[0])  # Use filename without extension as name

# Create a directory for unknown faces
unknown_faces_dir = 'unknown_faces/'
os.makedirs(unknown_faces_dir, exist_ok=True)

def log_attendance(student_name):
    student_id = get_student_id_by_name(student_name)
    if student_id:
        new_attendance = Attendance(student_id=student_id)
        db.session.add(new_attendance)
        db.session.commit()
        print(f"Logged attendance for {student_name} at {datetime.now()}")

def get_student_id_by_name(student_name):
    student = Student.query.filter_by(name=student_name).first()
    return student.student_id if student else None

# Start capturing video
camera.start_preview()
time.sleep(2)  # Allow time for the camera to initialize
video_capture = cv2.VideoCapture(0)  # Use 0 for the Pi camera

if not video_capture.isOpened():
    print("Failed to open video capture.")
    exit()  # Exit if the camera cannot be opened

while True:
    # Capture frame-by-frame
    ret, frame = video_capture.read()
    if not ret:
        print("Failed to capture frame.")
        break  # Exit the loop if frame capture fails

    # Convert the image from BGR to RGB
    rgb_frame = frame[:, :, ::-1]

    # Find all face locations and encodings in the current frame
    face_locations = face_recognition.face_locations(rgb_frame)
    face_encodings = face_recognition.face_encodings(rgb_frame, face_locations)

    for (top, right, bottom, left), face_encoding in zip(face_locations, face_encodings):
        # Check if the face is a known face
        matches = face_recognition.compare_faces(known_face_encodings, face_encoding)
        name = "Unknown"

        if True in matches:
            first_match_index = matches.index(True)
            name = known_face_names[first_match_index]

            # Log attendance in the database
            log_attendance(name)

        else:
            # Save the unknown face to the unknown_faces directory
            unknown_face_image = frame[top:bottom, left:right]
            unknown_face_filename = os.path.join(unknown_faces_dir, f"unknown_{datetime.now().strftime('%Y%m%d_%H%M%S')}.jpg")
            cv2.imwrite(unknown_face_filename, unknown_face_image)
            print(f"Saved unknown face to {unknown_face_filename}")

        # Draw a rectangle around the face and label it
        cv2.rectangle(frame, (left, top), (right, bottom), (255, 0, 0), 2)
        cv2.putText(frame, name, (left, top - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (255, 255, 255), 2)

    # Display the resulting frame
    cv2.imshow('Video Feed', frame)

    # Break the loop on 'q' key press
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the camera and close windows
video_capture.release()
cv2.destroyAllWindows()
