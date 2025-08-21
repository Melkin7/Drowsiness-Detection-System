import cv2
import numpy as np
import pyttsx3
import random
from keras.models import load_model
from keras.preprocessing.image import img_to_array
from playsound import playsound
from threading import Thread
from send_alert import send_sms_alert, make_call_alert  # Importing call function

def start_alarm(sound):
    """Play the alarm sound and speak a random safety command"""
    playsound(sound)
    commands = [
        "You are tired, take rest.",
        "Please pull over safely.",
        "Your safety is important.",
        "Stay alert, don't risk it.",
        "Close your eyes later, not now."
    ]
    engine = pyttsx3.init()
    engine.setProperty('rate', 150)
    engine.setProperty('volume', 1.0)
    message = random.choice(commands)
    engine.say(message)
    engine.runAndWait()

classes = ['Closed', 'Open']
face_cascade = cv2.CascadeClassifier("data/haarcascade_frontalface_default.xml")
left_eye_cascade = cv2.CascadeClassifier("data/haarcascade_lefteye_2splits.xml")
right_eye_cascade = cv2.CascadeClassifier("data/haarcascade_righteye_2splits.xml")
cap = cv2.VideoCapture(0)
model = load_model("drowiness_new7.h5")
count = 0
alarm_on = False
alarm_sound = "data/alarm.mp3"
status1 = ''
status2 = ''
status_text = "SAFE"
alert_sent = False  # To ensure alerts are sent only once per detection

while True:
    _, frame = cap.read()
    height = frame.shape[0]
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)

    for (x, y, w, h) in faces:
        cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 1)
        roi_gray = gray[y:y + h, x:x + w]
        roi_color = frame[y:y + h, x:x + w]
        left_eye = left_eye_cascade.detectMultiScale(roi_gray)
        right_eye = right_eye_cascade.detectMultiScale(roi_gray)

        for (x1, y1, w1, h1) in left_eye:
            cv2.rectangle(roi_color, (x1, y1), (x1 + w1, y1 + h1), (0, 255, 0), 1)
            eye1 = roi_color[y1:y1 + h1, x1:x1 + w1]
            eye1 = cv2.resize(eye1, (145, 145))
            eye1 = eye1.astype('float') / 255.0
            eye1 = img_to_array(eye1)
            eye1 = np.expand_dims(eye1, axis=0)
            pred1 = model.predict(eye1)
            status1 = np.argmax(pred1)
            break

        for (x2, y2, w2, h2) in right_eye:
            cv2.rectangle(roi_color, (x2, y2), (x2 + w2, y2 + h2), (0, 255, 0), 1)
            eye2 = roi_color[y2:y2 + h2, x2:x2 + w2]
            eye2 = cv2.resize(eye2, (145, 145))
            eye2 = eye2.astype('float') / 255.0
            eye2 = img_to_array(eye2)
            eye2 = np.expand_dims(eye2, axis=0)
            pred2 = model.predict(eye2)
            status2 = np.argmax(pred2)
            break

        if status1 == 2 and status2 == 2:
            count += 1
            status_text = "DROWSY"
            cv2.putText(frame, "Eyes Closed, Frame count: " + str(count), (10, 30),
                        cv2.FONT_HERSHEY_COMPLEX, 1, (0, 0, 255), 1)
            if count >= 10:
                status_text = "EMERGENCY"
                cv2.putText(frame, "Drowsiness Alert!!!", (100, height - 20),
                            cv2.FONT_HERSHEY_COMPLEX, 1, (0, 0, 255), 2)
                if not alarm_on:
                    alarm_on = True
                    t = Thread(target=start_alarm, args=(alarm_sound,))
                    t.daemon = True
                    t.start()

                    # 🚨 Send SMS and Call Alert
                    if not alert_sent:
                        alert_sent = True
                        sms_thread = Thread(target=send_sms_alert)
                        call_thread = Thread(target=make_call_alert)
                        sms_thread.daemon = True
                        call_thread.daemon = True
                        sms_thread.start()
                        call_thread.start()
        else:
            count = 0
            alarm_on = False
            alert_sent = False  # Reset for next detection
            status_text = "SAFE"
            cv2.putText(frame, "Eyes Open", (10, 30),
                        cv2.FONT_HERSHEY_COMPLEX, 1, (0, 255, 0), 1)

    cv2.putText(frame, f"STATUS: {status_text}", (10, height - 50),
                cv2.FONT_HERSHEY_SIMPLEX, 1.2,
                (0, 255, 0) if status_text == "SAFE" else
                (0, 255, 255) if status_text == "DROWSY" else
                (0, 0, 255), 3)

    cv2.imshow("Drowsiness Detector", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
