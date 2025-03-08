#!/usr/bin/env python3

import time
from picamera2 import Picamera2

def main():
    print("Initializing camera...")
    picam2 = Picamera2()
    
    # Configure the camera
    config = picam2.create_preview_configuration()
    picam2.configure(config)
    
    # Start the camera
    picam2.start()
    
    print("Camera initialized successfully!")
    print("Taking a test picture in 3 seconds...")
    time.sleep(3)
    
    # Capture an image
    metadata = picam2.capture_file("test_image.jpg")
    print(f"Image captured and saved as 'test_image.jpg'")
    print(f"Image metadata: {metadata}")
    
    # Stop the camera
    picam2.stop()
    print("Camera stopped. Test completed successfully!")

if __name__ == "__main__":
    main()
