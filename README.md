# HIL--Facial-Emotion-Recognition-System

Our project aims to monitor the driver closely to check whether he is fully awake or drowsy and check if he seems sad or angry. If the driver is feeling drowsy or out of the mood, the model is supposed to send a specific signal to the car, signaling it to park or get off of the road (supposing that it is a smart autonomous car).
We also care about the welfare of the driver, so we have worked on another idea which is checking the driver’s positive and negative emotions. If the driver is happy, sad, angry, or neutral, the car would customize the music that he is listening to according to his mood or provide multiple suggestions to places that would suit his current mood. For example, if the driver is feeling down, the car might look for near locations that might help boost his mood. 

## Repository Contents

  * Models (trials): This directory contains the training files of several models trained on the emotion recognition task. The weights can be downloaded.
  * Face Detection Models: This directory contains the weights and the helper files of the face detection models used in our application
  * FECNet: This directory contains the training files and the weights of the FEC model used in the classification stage 
  * VggFace Model: This directory contains the training files and the weights of the VggFace model used in the classification stage
  * Demos: Some demo files can be found to run the application on different platforms using the 2 models
  
## How to run the application demo?

  * Download the FECNet model weights from the link in the FECNet directory readme
  * To run the demo using Haar Cascade model, choose one of the following:
  
    1.  VggFace Model on PC: python3 demo_VggFace_HaarCascade_PC.py
    2.	VggFace Model on TX2: python3 demo_VggFace_HaarCascade_tx2.py
    3.	FECNet Model on PC: python3 demo_FECNet_HaarCascade_PC.py
    4.	FECNet Model on TX2: python3 demo_FECNet_HaarCascade_tx2.py
  * To run the demo using MTCNN model, go to the MTCNN directory, place a copy of both model weights to this path, and choose one of the following:
  
    1.	VggFace model: python3 demo_VggFace_mtcnn_jetsontx2.py
    2.	FECNet model: python3 demo_FEC_mtcnn_jetsontx2.py
