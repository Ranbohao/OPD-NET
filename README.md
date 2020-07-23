# OPD-NET
## Introduction

This is a tensorflow re-implementation of **OPD-Net: Prow Detection Based on Feature Enhancement and Improved Regression Model in Optical Remote Sensing Imagery**.

This project is modified based on [R2CNN_Faster-RCNN_Tensorflow](https://github.com/DetectionTeamUCAS/R2CNN_Faster-RCNN_Tensorflow), special thanks to the authors for sharing.

The complete code will be organized and made public later.

## Installation
### Requirements
1. tensorflow >= 1.2
2. cuda8/9.0
3. python3.6 (anaconda3 recommend)
4. opencv(cv2)
5. tfplot

### Compile
cd $PATH_ROOT/libs/box_utils/
python setup.py build_ext --inplace
cd $PATH_ROOT/libs/box_utils/cython_utils
python setup.py build_ext --inplace


## Trained Weights Preparation
1. please download [resnet50_v1](http://download.tensorflow.org/models/resnet_v1_50_2016_08_28.tar.gz)、[resnet101_v1](http://download.tensorflow.org/models/resnet_v1_101_2016_08_28.tar.gz) pre-trained models on Imagenet, put it to data/pretrained_weights.        
2. please download trained model by this project (we will upload later), and place it in the output/trained_weights.


## Inference
1. first modify the INFERENCE_IMAGE_PATH and INFERENCE_SAVE_PATH in the $PATH_ROOT/libs/configs/cfgs.py

2. in $PATH_ROOT/tools:
```python inference.py --gpu=0```


## TODO:
- [ ] data prepare
- [ ] Training
