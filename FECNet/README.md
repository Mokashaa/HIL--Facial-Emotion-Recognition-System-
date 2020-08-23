# FECNet

## Contents

* Implementation of FECNet descriebd in [1].

* FECNet for facial expressions recognition.

* FECNet for outliers removal applied on FER2013.

## FECNet Implementation

For implementing FECNet a pre-trained FaceNet-NN2 model is required. I didn't find any public weights for the architecture so I worked with a variation of FaceNet, which is the Inception-Resnet-V1 architecture. I used weights provided in [2]. I will refer to the variation I developed as FEC-IRNet.

## FECNet For Facial Expressions Recognition

After training FEC-IRNet, I used it as a base model in two classification models: An emotions classifier and a sleep classifier. I trained the emotions classifier on FER2013, and the sleep classifier on CEW.  
After training the two models I combined them in a unified model that detects emotions and drowsiness(sleep).  
saved model (Tensorflow 2.3): https://drive.google.com/file/d/1SMd_8myIWrkuSd_KscyrcYMJJ258jSyI/view?usp=sharing

## FECNet For Outliers Removal

FER2013 is dataset with a wide variety of facial expressions. It is one of the widely used datasets in this field. But it has the downsides of conataining images that may be falsely labeled or having exaggerated expressions. To filter the dataset from these images I tried the following approach:  

1. I used FEC-IRNet to map eaxh image in the dataset to 16-dimensional space.

2. I computed the mean and standard deviation of each emotion category.

3. I computed z-scores for each image in each class.

4. I removed the outliers using z-score; filter out if |z-score| >= 3.

5. I trained an emotions classifier on the filtered dataset.

## References

[1] Raviteja Vemulapalli and A. Agarwala, "A Compact Embedding for Facial Expression Similarity", CVPR 2019.  
[2] https://github.com/nyoki-mtl/keras-facenet.
