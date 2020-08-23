import cv2
import numpy as np
from tensorflow.keras.models import load_model

emotion_dict = {0: "angry", 1: "happy", 2: "neutral", 3: "sad"}
sleep_dict = {0: "alert", 1: "drowsy"}

model = load_model("./FECNet/FECNet_fer_no_outliers_20_8.hdf5")

cap = cv2.VideoCapture("nvarguscamerasrc ! video/x-raw(memory:NVMM), width=(int)1920, height=(int)1080,format=(string)NV12, framerate=(fraction)30/1 ! nvvidconv ! video/x-raw, format=(string)BGRx ! videoconvert ! video/x-raw, format=(string)BGR ! appsink")

i = 0
cv2.namedWindow("Driver assistant", cv2.WINDOW_NORMAL)
count = 1
sleep_prev = str("awake")

while True:

    ret, frame = cap.read()
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    face_cascade = cv2.CascadeClassifier('./Face_Detection_Models/Haar_Cascade_Classifier/haarcascade_frontalface_default.xml')
    faces = face_cascade.detectMultiScale(frame, 1.3, 5)

    for (x, y, w, h) in faces:
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 1)
        roi = frame[y:y + h, x:x + w]
        cropped_img = np.expand_dims(cv2.resize(roi, (160, 160)), 0)

        prediction = model.predict(cropped_img)
        sleep = sleep_dict[int(np.argmax(prediction[0]))]
        emotion = emotion_dict[int(np.argmax(prediction[1]))]

        if sleep == "drowsy" and sleep == sleep_prev:
            count = count+1
            sleep_prev = sleep
            if count > 10:
                cv2.rectangle(frame, (x + w + 20, y + 40), (x + w + 120, y + 55), (28, 13, 191), 13)
                cv2.putText(frame, "ALERT!!", (x + w + 32, y + 54), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 1,
                            cv2.LINE_AA)
        else:
            sleep_prev = sleep
            count = 1

        cv2.rectangle(frame, (x + w + 20, y + 5), (x + w + 120, y + 20), (206, 174, 17), 13)
        cv2.putText(frame, emotion, (x + w + 32, y + 18), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 1,
                    cv2.LINE_AA)

        if sleep == "awake":
            cv2.rectangle(frame, (x + w + 20, y + 40), (x + w + 120, y + 55), (206, 174, 17), 13)
            cv2.putText(frame, sleep, (x + w + 32, y + 53), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 1,
                        cv2.LINE_AA)
        elif sleep == "drowsy" and count <= 10:
            cv2.rectangle(frame, (x + w + 20, y + 40), (x + w + 120, y + 55), (14, 88, 235), 13)
            cv2.putText(frame, sleep, (x + w + 32, y + 53), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 1,
                        cv2.LINE_AA)

    cv2.imshow('Driver assistant', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
