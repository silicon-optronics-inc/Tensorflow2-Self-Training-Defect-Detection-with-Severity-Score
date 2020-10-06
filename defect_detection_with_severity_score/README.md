# Defect Detection Tool Guide
This document will go through how to use the models from [Object detection training guide](https://github.com/silicon-optronics-inc/Tensorflow2-Self-Training-Defect-Detection-With-Severity-Score/blob/master/object_detection/README.md) and [Classification training guide](https://github.com/silicon-optronics-inc/Tensorflow2-Self-Training-Defect-Detection-With-Severity-Score/blob/master/classification/README.md) to perform defect detection with severity score.

## Flow of Defect Detection 
To define own cropping and classification process, please modify the function 'predict_cls_img()' in './inference.py'.
![Flow Chart of_Defect Detection With Severity Score](https://github.com/silicon-optronics-inc/Object_detection_with_severity_score/blob/master/doc/Flow_Chart_of_Defect_Detection_With_Severity_Score.png)  

## Inference
Step 1. Copy pascal_label_map.pbtxt to current folder.  
Step 2. Open './config.ini' and set some parameters 
* img_path: Path to directory of image be detected.
* log_path: Path to predicted output folder.
* model_path: Path to models from [Object detection training guide](https://github.com/silicon-optronics-inc/Tensorflow2-Self-Training-Defect-Detection-With-Severity-Score/blob/master/object_detection/README.md) and [Classification training guide](https://github.com/silicon-optronics-inc/Tensorflow2-Self-Training-Defect-Detection-With-Severity-Score/blob/master/classification/README.md).
* obj_model_name: Object detection model name.
* obj_model_width: Object detection model width during training.
* obj_model_height:Object detection model height during training.
* obj_model_channel: Object detection model channel during training, usually is 3 if [convert_to_grayscale](https://github.com/tensorflow/models/blob/master/research/object_detection/protos/image_resizer.proto) is not set to True during training.
* cls_model_name: Classification model name.
* cls_model_width: Classification model width during training.
* cls_model_height: Classification model height during training.
* cls_model_channel: Classification model channel during training.
* threshold: Threshold of confidence score that objects will be shown when box_num is set to None.
* box_num: The number of objects to be displayed on the image.
* gpu_memory_fraction: A value between 0 and 1 that indicates what fraction of the available GPU memory to pre-allocate for each process.

Step 3. Run './run.bat', result will be saved under log_path just specified.


