# Driver_drowsiness_system_CNN

A real-time drowsiness detection system using CNN to monitor driver alertness.

## Features
- Real-time eye detection using OpenCV
- CNN model for drowsiness classification
- Audio alerts with voice commands
- SMS and call alerts via Twilio

## Setup
1. Clone the repository
2. Create virtual environment: `python -m venv tf-macos-env`
3. Activate: `source tf-macos-env/bin/activate`
4. Install dependencies: `pip install opencv-python keras tensorflow numpy pyttsx3 playsound twilio`
5. Configure Twilio: Copy `send_alert_template.py` to `send_alert.py` and add your credentials
6. Run: `python detect_drowsiness.py`

## Team Details
1. Adithya S
2. Melkin S
3. Nhowmitha S
