# Defect Detection With Severity Score
This project combines object detection and classification to detect defects and score them based on severity.

In the part of object detection, this project is modified based on tensorflow object detection api to make it more user-friendly and able to perform self-training (auto labeling).
As for the classification, the severity of the flaw is used as the classification standard, and the corresponding score weights are given to different categories during inference, making the application more flexible.

In addition, due to the differences in the image features and complexity of industrial defects and life scenes, the project does not use the built-in API model, but customizes the shallower model structure to avoid overfitting and improve the speed of training and inference. Guide to train and inference with custom model structure will also be mentioned.

![image](https://github.com/silicon-optronics-inc/Object_detection_with_severity_score/blob/master/doc/demo.gif)  

## Table of Contents
- [File Directory Description](#file-directory-description)
- [Getting Started](#getting-started)
  - [Requirements](#requirements)
  - [Installation](#installation)
  - [Suggestion](#suggestion)
- [Usage](#usage)
  - [LabelImg](#labelimg)
  - [Train](#train)
  - [Inference](#inference)

## File Directory Description
```
Tensorflow2-Self-Training-Defect-Detection-With-Severity-Score 
├── README.md
├── classification/
|   ├── README.md
│   ├── inference/
|   │   ├── config.ini
|   │   ├── inference.bat
|   │   └── inference.py
│   └── train/
|       ├── config.ini
|       ├── train.bat
|       ├── train.py
|       ├── loss/
|       |   └── focal_loss.py
|       └── model/
|           └── customizenet.py
|
├── defect_detection/
│   ├── README.md
│   ├── config.ini
│   ├── inference_v1.py
│   ├── object_label_map.pbtxt
│   └── run.bat
|
├── docs/
│   └── demo.gif
|
└── object_detection/
    ├── README.md
    ├── export_model/
    |   ├── export_inference_graph.bat
    |   ├── exporter_main_v2.py
    |   └── input/
    |       └── checkpoint
    ├── inference/
    |   ├── config.ini
    |   ├── inference_v1.py
    |   ├── object_label_map.pbtxt
    |   └── run.bat
    └── train/
        ├── run.bat
        ├── input/
        |   ├── 1_train_data/
        |   |   ├── img/
        |   |   ├── tfrecord/
        |   |   └── xml/
        |   ├── 2_eval_data/
        |   |   ├── img/
        |   |   ├── tfrecord/
        |   |   └── xml/
        |   ├── 3_unlabeled_data
        |   |   └── img/
        |   └── 4_setting/
        |       ├── faster_rcnn_resnet50_v1.config
        |       └── object_label_map.pbtxt
        └── tool/
            ├── auto_label.py
            ├── control_main.py
            ├── eval.py
            ├── generate_tfrecord_tf2.py
            ├── model_main_tf2.py
            └── xml_to_csv.py
```

## Getting Started
### Requirements
* CUDA
* cuDNN
* python 3.7
* tensorflow 2.3
* numpy 1.17.4
* object-detection 0.1

### Installation
Step1: Follow the official guide to install [Object Detection API with TensorFlow 2](https://github.com/tensorflow/models/blob/master/research/object_detection/g3doc/tf2.md)

Step2: Clone the repository
```git clone https://github.com/silicon-optronics-inc/Tensorflow2-Self-Training-Defect-Detection-With-Severity-Score.git```

### Suggestion
Modify some setting in C:\Python36\Lib\site-packages\object_detection\model_lib_v2.py (According to the installation path)
* Replace checkpoint_max_to_keep in function train_loop() with a bigger value to keep more number of models during training.
* Replace max_outputs in function eager_train_step() with 0 to avoid the summary events generated while training taking up too much disk space.

## Usage
### LabelImg
Before training object detection model, one must annotate objects in images and saved them as XML files in PASCAL VOC format. Tool LabelImg is recommended.  

[LabelImg github](https://github.com/tzutalin/labelImg)  
[LabelImg download link](https://www.dropbox.com/s/kqoxr10l3rkstqd/windows_v1.8.0.zip?dl=1)  


### Train
[Object detection training guide](https://github.com/silicon-optronics-inc/Tensorflow2-Self-Training-Defect-Detection-With-Severity-Score/blob/master/object_detection/README.md)  
[Classification training guide](https://github.com/silicon-optronics-inc/Tensorflow2-Self-Training-Defect-Detection-With-Severity-Score/blob/master/classification/README.md)  

### Inference
Step1: Copy object detection and classification models from [train](#train) to the folder where one want to place the models.  
Step2: Set the model path and some necessary parameters in the configuration for inference. For more detail, please refer to [defect detection README.md](https://github.com/silicon-optronics-inc/Tensorflow2-Self-Training-Defect-Detection-With-Severity-Score/blob/master/defect_detection/README.md)

