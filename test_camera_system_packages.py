#!/usr/bin/env python3

# Import necessary libraries
import time
from picamera2 import Picamera2
from datetime import datetime

def test_camera():
    print("Initializing PiCamera2...")
    
    # Initialize camera
    picam2 = Picamera2()
    
    # Configure the camera
    preview_config = picam2.create_preview_configuration()
    picam2.configure(preview_config)
    
    # Start the camera
    picam2.start()
    print("Camera started successfully!")
    
    # Wait for auto exposure and white balance to settle
    print("Waiting for camera to adjust...")
    time.sleep(2)
    
    # Take a picture
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"test_image_{timestamp}.jpg"
    picam2.capture_file(filename)
    print(f"Image captured: {filename}")
    
    # Stop the camera
    picam2.stop()
    print("Camera stopped")
    print("Test completed successfully!")

if __name__ == "__main__":
    test_camera()
