
# MobilenetV2 Implementations and Trials

## The following architectures were implemented and tested

### **1. MobilenetV2 as in** <https://arxiv.org/pdf/1801.04381.pdf> with 3 variations in the FC layers

python notebook: <https://colab.research.google.com/drive/16hXQ-2gIisd2gZSPOG-GWgsvr1cGW_yP>

saved models:

* mobilenetv2 no FC layers: <https://drive.google.com/open?id=10JVv-SCo3eV5pBXNAGrt4AI_iQI45sDk>

* mobilenetv2 one FC layer (1024 units): <https://drive.google.com/open?id=1o8hnemo8IFJ_C0KDEk3iiz8WJPXvsz24>

* mobilenetv2 one FC layer (2048 units): <https://drive.google.com/open?id=1gIxuyKIveS-8xkJXtlO3D48fTRlF1rY1>

evaluation notebook: <https://colab.research.google.com/drive/1b9bjDeYqNDqhfQslrMNhItqBWphtq-8x>

#### Feedback and notes

* The model was pre-trained on ImageNet dataset.

* Acheived test accuracies of 79.35%, 79.87%, 77.47% for mobilenetv2 without fc layer, with 1024 units fc layer and with 2048 units fc layer respectively.

* Small model sizes with around 3 million parameters.

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
