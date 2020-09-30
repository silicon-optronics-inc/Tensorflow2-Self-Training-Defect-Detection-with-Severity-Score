# Defect Detection With Severity Score
This project combines object detection and classification to detect defects and score them based on severity.

In the part of object detection, this project is modified based on tensorflow object detection api to make it more user-friendly and able to perform self-training (auto labeling).
As for the classification, the severity of the flaw is used as the classification standard, and the corresponding score weights are given to different categories during inference, making the application more flexible.

In addition, due to the differences in the image features and complexity of industrial defects and life scenes, the project does not use the built-in API model, but customizes the shallower model structure to avoid overfitting and improve the speed of training and inference. Guide to train and inference with custom model structure will also be mentioned.

![image](https://github.com/silicon-optronics-inc/Object_detection_with_severity_score/blob/master/doc/demo.gif)  

## Table of Contents
- [Getting Started](#getting-started)
  - [Requirements](#requirements)
  - [Installation](#installation)
## Getting Started
### Requirements
* python 3.7
* tensorflow 2.3
* object-detection 0.1
* numpy 1.17.4

### Installation
