import cv2

# Initialize the camera
video_capture = cv2.VideoCapture(0)  # Use 0 for the Pi camera

if not video_capture.isOpened():
    print("Failed to open camera.")
else:
    print("Camera opened successfully.")

while True:
    ret, frame = video_capture.read()
    if not ret:
        print("Failed to capture frame.")
        break

    cv2.imshow('Test Video Feed', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

video_capture.release()
cv2.destroyAllWindows()

