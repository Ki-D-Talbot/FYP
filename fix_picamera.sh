#!/bin/bash

# Create a fresh virtual environment
echo "Creating a fresh virtual environment..."
cd ~/Documents/FYP
rm -rf .venv
python3 -m venv .venv
source .venv/bin/activate

# Install the correct versions of packages
echo "Installing compatible versions of packages..."
pip install numpy==1.24.3  # Using an older version that's compatible with picamera2

# Install PiCamera2 system-wide dependencies if they aren't already
echo "Ensuring system dependencies are installed..."
sudo apt update
sudo apt install -y python3-picamera2 python3-libcamera python3-kms++

# Create a simple test script
echo "Creating a test script..."
cat > test_picamera2.py << 'EOL'
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
EOL

chmod +x test_picamera2.py

echo "----------------------------------------------"
echo "Setup completed. To test the camera, run:"
echo "source .venv/bin/activate"
echo "python test_picamera2.py"
echo "----------------------------------------------"
#!/bin/bash

# Navigate to your project directory
cd ~/Documents/FYP

# Remove existing virtual environment if it exists
if [ -d ".venv" ]; then
  echo "Removing existing virtual environment..."
  rm -rf .venv
fi

# Create new virtual environment with access to system packages
echo "Creating new virtual environment with system packages access..."
python3 -m venv .venv --system-site-packages

# Activate the new environment
echo "Activating virtual environment..."
source .venv/bin/activate

# Create a simple test script
echo "Creating a test script..."
cat > test_camera_system_packages.py << 'EOL'
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
EOL

chmod +x test_camera_system_packages.py

echo ""
echo "========================================================"
echo "Setup completed successfully!"
echo ""
echo "Your new virtual environment has access to system packages,"
echo "including PiCamera2 and other libraries installed via apt."
echo ""
echo "To test the camera, run:"
echo "source .venv/bin/activate"
echo "python test_camera_system_packages.py"
echo "========================================================"
