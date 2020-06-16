from keras.models import load_model
import cv2
import numpy as np

emotion_dict = {0: "Angry", 1: "Happy", 2: "neutral", 3: "sad"}
sleep_dict = {0: "alert", 1: "drowsy"}

model = load_model("sleep_emotion_model.hdf5")

cap = cv2.VideoCapture(0)
i = 0
cv2.namedWindow("window", cv2.WINDOW_NORMAL)

while True:

    ret, frame = cap.read()
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
    faces = face_cascade.detectMultiScale(frame, 1.3, 5)

    for (x, y, w, h) in faces:
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 1)
        roi = frame[y:y + h, x:x + w]
        cropped_img = np.expand_dims(cv2.resize(roi, (224, 224)), 0)

        prediction = model.predict(cropped_img)
        sleep = prediction[0]
        emotion = prediction[1]
        cv2.putText(frame, emotion_dict[int(np.argmax(emotion))] + ", " + sleep_dict[int(np.argmax(sleep))], (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 1, cv2.LINE_AA)

    cv2.imshow('window', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
