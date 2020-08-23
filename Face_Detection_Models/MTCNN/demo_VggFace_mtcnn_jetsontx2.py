"""trt_mtcnn.py

This script demonstrates how to do real-time face detection with
Cython wrapped TensorRT optimized MTCNN engine.
"""

import sys
import time
import argparse

import cv2
from utils.camera import add_camera_args, Camera
from utils.display import open_window, set_display, show_fps
from utils.mtcnn import TrtMtcnn
from keras.models import load_model
import numpy as np

emotion_dict = {0: 'angry', 1: 'happy', 2: 'neutral', 3: 'sad'}
sleep_dict = {0: "awake", 1: "drowsy"}

model = load_model("./whole_model.hdf5")
WINDOW_NAME = 'TrtMtcnnDemo'
BBOX_COLOR = (0, 255, 0)  # green
count = 1
sleep_prev = str("awake")

def parse_args():
    """Parse input arguments."""
    desc = ('Capture and display live camera video, while doing '
            'real-time face detection with TrtMtcnn on Jetson '
            'Nano')
    parser = argparse.ArgumentParser(description=desc)
    parser = add_camera_args(parser)
    parser.add_argument('--minsize', type=int, default=40,
                        help='minsize (in pixels) for detection [40]')
    args = parser.parse_args()
    return args


def show_faces(img, boxes, landmarks):
    """Draw bounding boxes and face landmarks on image."""
    global sleep_prev
    global count
    img1 = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    for bb, ll in zip(boxes, landmarks):
        x1, y1, x2, y2 = int(bb[0]), int(bb[1]), int(bb[2]), int(bb[3])
        #################################
        roi_color = img1[y1:y2, x1:x2, :]
        cropped_img = np.expand_dims(cv2.resize(roi_color, (224, 224)), 0)
        prediction2 = model.predict(cropped_img)
        sleep = prediction2[0]
        emotion = prediction2[1]
        print(emotion_dict[int(np.argmax(emotion))] , "\n" , sleep_dict[int(np.argmax(sleep))] )
        ################################
        cv2.rectangle(img, (x1, y1), (x2, y2), (206, 174, 17), 1)
        sleep = sleep_dict[int(np.argmax(prediction2[0]))]
        emotion = emotion_dict[int(np.argmax(prediction2[1]))]

        if sleep == "drowsy" and sleep == sleep_prev:
            count = count+1
            sleep_prev = sleep
            if count > 10:
                cv2.rectangle(img, (x2 + 20, y1+ 40), (x2 + 120, y1 + 55), (28, 13, 191), 13)
                cv2.putText(img, "ALERT!!", (x2 + 32, y1 + 54), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 1,
                            cv2.LINE_AA)
        else:
            sleep_prev = sleep
            count = 1

        cv2.rectangle(img, (x2 + 20, y1 + 5), (x2 + 120, y1 + 20), (206, 174, 17), 13)
        cv2.putText(img, emotion, (x2 + 32, y1 + 18), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 1,
                    cv2.LINE_AA)

        if sleep == "awake":
            cv2.rectangle(img, (x2 + 20, y1 + 40), (x2+ 120, y1 + 55), (206, 174, 17), 13)
            cv2.putText(img, sleep, (x2 + 32, y1 + 53), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 1,
                        cv2.LINE_AA)
        elif sleep == "drowsy" and count <= 10:
            cv2.rectangle(img, (x2 + 20, y1 + 40), (x2 + 120, y1 + 55), (14, 88, 235), 13)
            cv2.putText(img, sleep, (x2 + 32, y1 + 53), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 1,
                        cv2.LINE_AA)
    return img


def loop_and_detect(cam, mtcnn, minsize):
    """Continuously capture images from camera and do face detection."""
    full_scrn = False
    fps = 0.0
    tic = time.time()
    while True:
        if cv2.getWindowProperty(WINDOW_NAME, 0) < 0:
            break
        img = cam.read()
        if img is not None:
            dets, landmarks = mtcnn.detect(img, minsize=minsize)
            print('{} face(s) found'.format(len(dets)))
            img = show_faces(img, dets, landmarks)
            img = show_fps(img, fps)
            cv2.imshow(WINDOW_NAME, img)
            toc = time.time()
            curr_fps = 1.0 / (toc - tic)
            # calculate an exponentially decaying average of fps number
            fps = curr_fps if fps == 0.0 else (fps*0.95 + curr_fps*0.05)
            tic = toc
        key = cv2.waitKey(1)
        if key == 27:  # ESC key: quit program
            break
        elif key == ord('F') or key == ord('f'):  # Toggle fullscreen
            full_scrn = not full_scrn
            set_display(WINDOW_NAME, full_scrn)


def main():
    args = parse_args()
    cam = Camera(args)
    cam.open()
    if not cam.is_opened:
        sys.exit('Failed to open camera!')

    mtcnn = TrtMtcnn()

    cam.start()
    open_window(WINDOW_NAME, args.image_width, args.image_height,
                'Camera TensorRT MTCNN Demo for Jetson TX2')
    loop_and_detect(cam, mtcnn, args.minsize)

    cam.stop()
    cam.release()
    cv2.destroyAllWindows()

    del(mtcnn)


if __name__ == '__main__':
    main()
