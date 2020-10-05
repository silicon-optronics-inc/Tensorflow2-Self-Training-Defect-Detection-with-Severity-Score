# Object Detection Training Guide
This document will explain how to use these programs to train the model and the parameters that need to be adjusted.

## Data Preparation
Four types of files are needed. First two could be get if follow the instructions of [LabelImg github](https://github.com/tzutalin/labelImg). The last two files could be found under 'object_detection/train/input/4_setting/'.
1. Images
2. XML, follow PASCAL VOC format.
3. pascal_label_map.pbtxt, contains the id and name of the object to be detected. THe name and number of items should be identical to those in xml.
4. Config, configure the object detection training pipeline. One can define their own training process by modifying or replacing it with any configuration file in other online resources.

After preparing the required files above, copy images and XML to corresponding folder(img and xml) under 'object_detection/train/input/1_train_data/' and 'object_detection/train/input/2_eval_data/' respectively as training and evaluation data. 
To enable self-training, unlabeled images should be put under 'object_detection/train/input/3_unlabeled_data/img/'


## Basic Parameters Setting
### object_detection/train/run.bat
* mode: Types of training mode. {0: Training without evaluating, 1: Training with gpu and evaluating with cpu, 2: Training with gpu, evaluating with cpu, and auto-label while meet the requirement, 3: Evaluate all models}
* pipeline_config: Name of config file.

### object_detection/train/input/4_setting/config
* num_classes: The number of object types to be detected.
* fixed_shape_resizer: The height and width of model you want.
* train_input_reader\tf_record_input_reader\input_path: Path to training tfrecord. '.record-?????-of-00010' should be added since sharded tfrecord are used. For example, "D:\\object_detection\\train\\input\\1_train_data\\tfrecord\\1_train_data.record-?????-of-00010". 
* eval_input_reader\tf_record_input_reader\input_path: Path to evaluating tfrecord. '.record-?????-of-00010' should be added.
* label_map_path: Path to pascal_label_map.pbtxt.


## Advanced Parameters Setting (Optional)
### object_detection/train/tool/control_main.py
* first_env["CUDA_VISIBLE_DEVICES"]: Main hardware setting used for training. '0' means use the first gpu, and multi-gpu can be used by setting the parameter to '0, 1, ...'
* second_env["CUDA_VISIBLE_DEVICES"]: Hardware setting for evaluating while training. '-1' means use cpu.
* eval_index: Evaluation metric used as monitoring indicator to start auto-labeling. More detail can be found on [COCO evaluation metrics](https://cocodataset.org/#detection-eval).
* eval_threshold: Requirement threshold of evaluation metric to start auto-labeling.
* IOU_threshold: The IOU threshold used to determine the confidence score threshold of each object during automatic labeling.

## Start Training
Double click object_detection/train/run.bat.

## Export Model
After finishing training, copy  
1. pascal_label_map.pbtxt,  
2. Config,  
3. the model wanted under 'object_detection/train/train/',  
to '/object_detection/export_model/input/',  
then edit the model step in '/object_detection/export_model/input/checkpoint'.  
Double click '/object_detection/export_model/export_inference_graph.bat'.
