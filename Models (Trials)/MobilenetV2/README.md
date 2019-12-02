
# MobilenetV2 Implementations and Trials

## The following architectures were implemented and tested

### **1. MobilenetV2 as in** <https://arxiv.org/pdf/1801.04381.pdf>

python notebook: <https://colab.research.google.com/drive/1XSM0jSrHe3YxKqO-vvHY1xu324auD95V>

saved model: <https://drive.google.com/open?id=1gIxuyKIveS-8xkJXtlO3D48fTRlF1rY1>

#### Feedback and notes

* The model was pre-trained on ImageNet dataset.

* Acheived test accuracy of 75% after 8 epochs of training.

* Small model size with around 2.5 million parameters.

* Further training (more than 8 epochs) would improve performance.

### **2. MobilenetV2 variant (mobilenetv2 small) as in** <https://arxiv.org/pdf/1902.05411.pdf>

python notebook: <https://colab.research.google.com/drive/1tfWn1fDfampt5bglujqYoW7PKawy1J2d>

saved model: <https://drive.google.com/open?id=1-29VHoeCSoKwlyOhun-wzxu996fBcTNb>

#### Feedback and notes

* The model was trained from scratch.

* Acheived test accuracy of 40% after 8 epochs of training.

* Extra small model size with around 700 thousend parameters.

* Further training (more than 8 epochs) should improve performance.

* Reported accuracy is 80%.

### 3. **Spatial Transformer Layer as in** <https://arxiv.org/abs/1506.02025>

python notebook: <https://colab.research.google.com/drive/1NzTPbM5Hm53tykgz15wTNqCwtEvKu9YE>

#### Feedback and notes

* (to-do)
