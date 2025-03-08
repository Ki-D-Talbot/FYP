#!/usr/bin/env python3
import numpy as np
try:
    from picamera2 import Picamera2
    print("picamera2 imported successfully.")
    picam2 = Picamera2()
    print("Picamera2 initialised.")
    config = picam2.create_preview_configuration()
    picam2.configure(config)
    picam2.start()
    print("Camera started.")
    frame = picam2.capture_array()
    print("Frame captured as NumPy array.")
    print(f"NumPy array shape: {frame.shape}")
    print(f"NumPy array dtype: {frame.dtype}")
    picam2.stop()
    print("Camera stopped.")

except ImportError as e:
    print(f"ImportError: {e}")
except ModuleNotFoundError as e:
    print(f"ModuleNotFoundError: {e}")
except Exception as e:
    print(f"An unexpected error occurred: {e}")
else:
    print("\\nTest completed successfully: NumPy and picamera2 are working within the virtual environment.")

