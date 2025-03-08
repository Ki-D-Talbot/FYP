#!/usr/bin/env python3

import time
import sys
import os

print("PiCamera2 Clean Test")
print("=====================")

try:
    # Import picamera2 for Pi Camera Module
    from picamera2 import Picamera2
    import cv2
    
    print("Attempting clean camera initialization...")
    
    # Make sure we don't have lingering instances
    try:
        os.system("sudo pkill -f 'python.*libcamera'")
        time.sleep(1)
    except:
        pass
    
    # Initialize camera with explicit configuration
    picam2 = Picamera2()
    
    # Print camera information
    print("Camera detected:")
    print(f"  Camera ID: {picam2.camera_properties['Id']}")
    print(f"  Model: {picam2.camera_properties.get('Model', 'Unknown')}")
    print(f"  Location: {picam2.camera_properties.get('Location', 'Unknown')}")
    
    # Create very basic configuration
    config = picam2.create_still_configuration()
    picam2.configure(config)
    
    print("Camera configured, starting...")
    picam2.start()
    
    print("Waiting for camera to initialize...")
    time.sleep(2)
    
    print("Capturing single image...")
    frame = picam2.capture_array()
    print(f"Captured image with shape: {frame.shape}")
    
    # Save image
    filename = "clean_camera_test.jpg"
    cv2.imwrite(filename, cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
    print(f"Image saved as {filename}")
    
    # Properly close the camera
    print("Stopping camera...")
    picam2.stop()
    
    print("Test completed successfully!")
    
except Exception as e:
    print(f"Error: {e}")
    print(f"Error type: {type(e).__name__}")
    print("Test failed.")

print("=====================")
