import numpy as np
from keras.engine import Model
from keras.models import load_model, Sequential
from PIL import Image

VGGFace_SleepModel = load_model("F:/Engineering/Graduation Project/SleepAndEmotion/Models_Weights/model_vggface_SleepV3.hdf5")
VGGFace_SleepModel.load_weights("F:/Engineering/Graduation Project/SleepAndEmotion/Models_Weights/weights_vggface_SleepV3.14-0.86.hdf5")
VGGFace_EmotionModel = load_model("F:/Engineering/Graduation Project/SleepAndEmotion/Models_Weights/model_vggface_emotion.hdf5")
VGGFace_EmotionModel.load_weights("F:/Engineering/Graduation Project/SleepAndEmotion/Models_Weights/weights_vggface_emotion.20-0.84.hdf5")

sleep_model = Sequential()
for layer in VGGFace_SleepModel.layers[40:]:
    sleep_model.model.add(layer)
sleep_model.build((None,28,28,128))
#sleep_model.summary()

emotion_model = Sequential()
for layer in VGGFace_EmotionModel.layers[100:]:
    emotion_model.model.add(layer)
emotion_model.build((None,14,14,1024))
#emotion_model.summary()

base_1 = VGGFace_EmotionModel.layers[39].output #output of VGGFace_EmotionModel.layers[39] is the same output of VGGFace_SleepModel.layers[39];
                                                #Because first 40 layers in both models are freezed (non-trainable), therefore they have the same weights
                                                #This is done to avoid Error: Graph disconnected
base_2 = VGGFace_EmotionModel.layers[99].output
sleep_op = sleep_model(base_1)
emotion_op = emotion_model(base_2)

sleep_emotion_model = Model(inputs=VGGFace_EmotionModel.input, outputs=[sleep_op, emotion_op], name="sleep_emotion_model")
sleep_emotion_model.summary()
sleep_emotion_model.save("sleep_emotion_model.hdf5")

'''
img = Image.open("F:/Engineering/Graduation Project/SleepAndEmotion/FaceDataset/Sleep/closed_eye_0030.jpg_face_2.jpg")
img = img.resize((224, 224))
img = np.array(img)
img = np.expand_dims(img, 0)
predict = whole_model.predict(img)
print(predict)

predict_s = VGGFace_SleepModel.predict(img)
print(int(np.argmax(predict_s)))

predict_e = VGGFace_EmotionModel.predict(img)
print(int(np.argmax(predict_e)))
'''