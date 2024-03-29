# Defect Detection With Severity Score
This project combines object detection to detect defects and classification to score them based on the severity.

In the part of object detection, this project is modified based on [tensorflow object detection api](https://github.com/tensorflow/models/blob/master/research/object_detection/g3doc/tf2.md) to make it more user-friendly and able to perform self-training (auto labeling).

As for classification, [keras](https://keras.io/api/) is used. When training the model, this project divides flaws into several categories according to the severity of them. Then the corresponding score weights(the more serious the higher the weight) are given to different categories during inference to get the severity score, making the application more flexible. For example, if the score weights are [0.2, 0.4, 0.6, 0.8, 1], assuming that flaw 1 gets scores [0, 0.02, 0.08, 0.85, 0.05] for each category, then the final severity score will be 0.2 * 0 + 0.4 * 0.02 + ... + 1 * 0.05 = 0.786

In addition, due to the differences in the image features and complexity of industrial defects and life scenes, the project does not use the built-in API model, but customizes the shallower model structure to avoid overfitting and improve the speed of training and inference. Guide to train and inference with custom model structure will also be mentioned.

![image](https://github.com/silicon-optronics-inc/Object_detection_with_severity_score/blob/master/doc/demo.gif)  

## Table of Contents
- [File Directory Description](#file-directory-description)
- [Flow of Defect Detection](#flow-of-defect-detection)
- [Getting Started](#getting-started)
  - [Requirements](#requirements)
  - [Installation](#installation)
  - [Suggestion](#suggestion)
- [Usage](#usage)
  - [LabelImg](#labelimg)
  - [Train](#train)
  - [Inference](#inference)
- [Acknowledgements](#acknowledgements)
- [About SOI](#about-soi)

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

## Flow of Defect Detection 
This project first uses object detection to find out the coordinates and types of defects, and then crops the defects according to the type and sends them to the corresponding classification model in the next step, and let the classification model give scores according to the severity of the defects, and finally mark the type, severity and location of each defect on the picture.  

Some of the defects that do not care about the severity can skip the classification model and directly use the object detection score.  


![Flow Chart of_Defect Detection With Severity Score](https://github.com/silicon-optronics-inc/Object_detection_with_severity_score/blob/master/doc/Flow_Chart_of_Defect_Detection_With_Severity_Score.png)  


## Getting Started
### Requirements
* CUDA
* cuDNN
* python 3.7
* tensorflow 2.3
* numpy 1.17.4
* object-detection 0.1

### Installation
Step1: Follow the official guide to install [Object Detection API with TensorFlow 2](https://github.com/tensorflow/models/blob/master/research/object_detection/g3doc/tf2.md).  
Step2: Clone the repository  
```git clone https://github.com/silicon-optronics-inc/Tensorflow2-Self-Training-Defect-Detection-With-Severity-Score.git```

### Suggestion
Modify some setting in model_lib_v2.py (Usually under C:\Python36\Lib\site-packages\object_detection\).
* Replace checkpoint_max_to_keep in function train_loop() with a bigger value to keep more number of models during training.
* Replace max_outputs in function eager_train_step() with 0 to avoid the summary events generated while training taking up too much disk space.

## Usage
### LabelImg
Before starting training the object detection model, the objects in the images must be annotated and saved as XML files in PASCAL VOC format. Tool LabelImg is recommended.  
[LabelImg github](https://github.com/tzutalin/labelImg)  
[LabelImg download link](https://www.dropbox.com/s/kqoxr10l3rkstqd/windows_v1.8.0.zip?dl=1)  


### Train
Because we use the output of the first-stage object detection as the input of the second-stage classification model, and the input of the classification model must be a fixed size; hence, when training the second-stage classification model to get the severity score, please crop fixed-size defect features from images to use as training data for classification models.  

A classification model can only score the severity of one defect. In other words, the number of trained classification models must be equal to the number of defect types that you want to give a severity score. 
For how to train the model, please refer to  
[Object detection training guide](https://github.com/silicon-optronics-inc/Tensorflow2-Self-Training-Defect-Detection-With-Severity-Score/blob/master/object_detection/README.md)  
[Classification training guide](https://github.com/silicon-optronics-inc/Tensorflow2-Self-Training-Defect-Detection-With-Severity-Score/blob/master/classification/README.md)  

### Inference
Step1: Copy object detection and classification models from [train](#train) to the folder where one want to place the models.  
Step2: Open './defect_detection_with_severity_score/', set the model path and some necessary parameters in the configuration for inference. For more information, please refer to [defect detection README.md](https://github.com/silicon-optronics-inc/Tensorflow2-Self-Training-Defect-Detection-With-Severity-Score/blob/master/defect_detection/README.md).

## Acknowledgements
[AWS Automate Data Labeling](https://docs.aws.amazon.com/sagemaker/latest/dg/sms-automated-labeling.html)  
[Google object detection api](https://github.com/tensorflow/models/tree/master/research/object_detection)  
[LabelImg](https://github.com/tzutalin/labelImg)  
[TensorFlow-Object-Detection-API-Tutorial-Train-Multiple-Objects-Windows-10](https://github.com/EdjeElectronics/TensorFlow-Object-Detection-API-Tutorial-Train-Multiple-Objects-Windows-10)  
[TensorFlow 2 Object Detection API tutorial](https://tensorflow-object-detection-api-tutorial.readthedocs.io/en/latest/)  
 


## About SOI
This project is developed by [SOI](http://www.soinc.com.tw/en/).  
[SOI](http://www.soinc.com.tw/en/)(Silicon Optronics, Inc.) is a CMOS image sensor design company founded in May 2004 and listed in the Taiwan Emerging Market under ticker number 3530. The management team is comprised of seasoned executives with extensive experience in pixel design, image processing algorithms, analog circuits, optoelectronic systems, supply chain management, and sales.
