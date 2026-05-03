import cv2
import mediapipe as mp
import numpy as np
import pyttsx3
import time
import threading
import winsound   # For beep (Windows)

# 🔊 Voice setup
engine = pyttsx3.init()
engine.setProperty('rate', 160)

# Flags
alarm_on = False
voice_on = False

# 🔊 Voice function (non-blocking)
def speak_alert():
    global voice_on
    voice_on = True
    engine.say("Drowsiness detected. Please open your eyes")
    engine.runAndWait()
    voice_on = False

# 🔊 Continuous Beep
def beep_sound():
    global alarm_on
    while alarm_on:
        winsound.Beep(1000, 500)  # frequency, duration

# MediaPipe setup
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(refine_landmarks=False, max_num_faces=1)

LEFT_EYE = [33, 160, 158, 133, 153, 144]
RIGHT_EYE = [362, 385, 387, 263, 373, 380]

def eyeAspectRatio(eye):
    A = np.linalg.norm(eye[1] - eye[5])
    B = np.linalg.norm(eye[2] - eye[4])
    C = np.linalg.norm(eye[0] - eye[3])
    return (A + B) / (2.0 * C)

# 🔥 Thresholds
earThresh = 0.21
close_start_time = None

cam = cv2.VideoCapture(0, cv2.CAP_DSHOW)

while True:
    ret, frame = cam.read()
    if not ret:
        break

    frame = cv2.resize(frame, (320, 240))
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    result = face_mesh.process(rgb)

    if result.multi_face_landmarks:
        for face_landmarks in result.multi_face_landmarks:
            h, w, _ = frame.shape

            leftEye = []
            rightEye = []

            for i in LEFT_EYE:
                x = int(face_landmarks.landmark[i].x * w)
                y = int(face_landmarks.landmark[i].y * h)
                leftEye.append([x, y])

            for i in RIGHT_EYE:
                x = int(face_landmarks.landmark[i].x * w)
                y = int(face_landmarks.landmark[i].y * h)
                rightEye.append([x, y])

            leftEye = np.array(leftEye)
            rightEye = np.array(rightEye)

            ear = (eyeAspectRatio(leftEye) + eyeAspectRatio(rightEye)) / 2.0

            # Draw eyes
            cv2.polylines(frame, [leftEye], True, (0, 255, 0), 1)
            cv2.polylines(frame, [rightEye], True, (0, 255, 0), 1)

            if ear < earThresh:
                if close_start_time is None:
                    close_start_time = time.time()

                elapsed = time.time() - close_start_time

                if elapsed > 3:
                    cv2.putText(frame, "DROWSINESS ALERT!", (10, 30),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

                    # 🔊 Start Beep
                    if not alarm_on:
                        alarm_on = True
                        threading.Thread(target=beep_sound, daemon=True).start()

                    # 🗣️ Speak once (non-blocking)
                    if not voice_on:
                        threading.Thread(target=speak_alert, daemon=True).start()

            else:
                close_start_time = None
                alarm_on = False  # Stop beep

    cv2.imshow("Drowsiness Detector", frame)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cam.release()
cv2.destroyAllWindows()
