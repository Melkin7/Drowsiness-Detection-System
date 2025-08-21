from flask import Flask, render_template, Response
import cv2
import numpy as np
from keras.models import load_model
from keras.preprocessing.image import img_to_array
from playsound import playsound
from threading import Thread
import time

app = Flask(__name__)
model = load_model("drowiness_new7.h5")
face_cascade = cv2.CascadeClassifier("data/haarcascade_frontalface_default.xml")
left_eye_cascade = cv2.CascadeClassifier("data/haarcascade_lefteye_2splits.xml")
right_eye_cascade = cv2.CascadeClassifier("data/haarcascade_righteye_2splits.xml")

status1 = ''
status2 = ''
classes = ['Closed', 'Open']

# Alarm sound
def start_alarm(sound="data/alarm.mp3"):
    playsound(sound)

def gen_frames():
    global status1, status2
    cap = cv2.VideoCapture(0)
    
    closed_start_time = None
    alert_triggered = False

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        height = frame.shape[0]
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, 1.3, 5)

        for (x, y, w, h) in faces:
            roi_gray = gray[y:y+h, x:x+w]
            roi_color = frame[y:y+h, x:x+w]

            left_eye = left_eye_cascade.detectMultiScale(roi_gray)
            right_eye = right_eye_cascade.detectMultiScale(roi_gray)

            for (x1, y1, w1, h1) in left_eye:
                eye = roi_color[y1:y1+h1, x1:x1+w1]
                eye = cv2.resize(eye, (145, 145))
                eye = eye.astype('float') / 255.0
                eye = img_to_array(eye)
                eye = np.expand_dims(eye, axis=0)
                pred = model.predict(eye)
                status1 = np.argmax(pred)
                break

            for (x2, y2, w2, h2) in right_eye:
                eye = roi_color[y2:y2+h2, x2:x2+w2]
                eye = cv2.resize(eye, (145, 145))
                eye = eye.astype('float') / 255.0
                eye = img_to_array(eye)
                eye = np.expand_dims(eye, axis=0)
                pred = model.predict(eye)
                status2 = np.argmax(pred)
                break

            # Both eyes closed
            if status1 == 0 and status2 == 0:
                if closed_start_time is None:
                    closed_start_time = time.time()
                else:
                    elapsed_time = time.time() - closed_start_time
                    cv2.putText(frame, f"Eyes Closed for {int(elapsed_time)}s", (10, 30),
                                cv2.FONT_HERSHEY_COMPLEX, 1, (0, 0, 255), 1)

                    if elapsed_time >= 3 and not alert_triggered:
                        cv2.putText(frame, "EMERGENCY ALERT!", (100, height - 20),
                                    cv2.FONT_HERSHEY_COMPLEX, 1, (0, 0, 255), 2)
                        alert_triggered = True
                        Thread(target=start_alarm, daemon=True).start()
            else:
                closed_start_time = None
                alert_triggered = False
                cv2.putText(frame, "Eyes Open", (10, 30),
                            cv2.FONT_HERSHEY_COMPLEX, 1, (0, 255, 0), 1)

        ret, buffer = cv2.imencode('.jpg', frame)
        frame = buffer.tobytes()
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

    cap.release()

@app.route('/')
def index():
    return render_template('index.html')  # This renders the main page

@app.route('/start', methods=['POST'])
def start_detection():
    return render_template('index.html', video_feed=True)  # Start detection on POST

@app.route('/video_feed')
def video_feed():
    return Response(gen_frames(),
                    mimetype='multipart/x-mixed-replace; boundary=frame')

if __name__ == '__main__':
    app.run(debug=True)
