from tensorflow.keras.models import load_model
import cv2 
import numpy as np

emotion_dict = {0: "Angry", 1: "Fear", 2: "Happy", 3: "Neutral", 4: "Sad", 5: "Surprise"}

model = load_model("./mobilenetv2_1024")

cap = cv2.VideoCapture(0)
i = 0
cv2.namedWindow("window", cv2.WINDOW_NORMAL)
# cv2.setWindowProperty("window",cv2.WND_PROP_FULLSCREEN,cv2.WINDOW_FULLSCREEN)

while cap.isOpened():
    
    ret, frame = cap.read()
    frame = cv2.flip(frame, 1)

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    color = cv2.cvtColor(gray, cv2.COLOR_GRAY2RGB)

    face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)

    for (x, y, w, h) in faces:
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 1)
        roi_color = color[y:y + h, x:x + w, :]
        # print(roi_color.shape)
        cropped_img = np.expand_dims(cv2.resize(roi_color, (96, 96)), 0)
        # print(cropped_img.shape)
        cv2.normalize(cropped_img, cropped_img, alpha=0, beta=1, norm_type=cv2.NORM_L2, dtype=cv2.CV_32F)
        # print(cropped_img.shape)
       
        # prediction1 = model.predict(cropped_img/255)
        prediction2 = model.predict(cropped_img/255)
        # print(prediction2)
        # both = np.hstack([prediction2, prediction1])
        # final_prediction = regr.predict_proba(both)
        cv2.putText(frame, emotion_dict[int(np.argmax(prediction2))], (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 1, cv2.LINE_AA)
        # print('hi')

    # print('hi')
    cv2.imshow('window', frame)
    # print('hi')
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()