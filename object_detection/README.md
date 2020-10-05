# Object Detection Tool Guide
This document will go through how to train the object detection model and use it for inference.

## Data Preparation
Four types of files are needed. First two could be get through followomg the instructions of [LabelImg github](https://github.com/tzutalin/labelImg). The last two files could be found under 'train/input/4_setting/'.
1. Images
2. XML, follow PASCAL VOC format.
3. pascal_label_map.pbtxt, contains the id and name of the object to be detected. THe name and number of items should be identical to those in xml.
4. Config, configure the object detection training pipeline. One can define their own training process by modifying or replacing it with any configuration file in other online resources.

After preparing the required files above, copy images and XML to corresponding folder(img and xml) under 'train/input/1_train_data/' and 'train/input/2_eval_data/' respectively as training and evaluation data. 
To enable self-training, unlabeled images should be put under 'train/input/3_unlabeled_data/img/'


## Parameters
### Basic Parameters Setting
#### train/run.bat
* mode: Types of training mode. {0: Training without evaluating, 1: Training with gpu and evaluating with cpu, 2: Training with gpu, evaluating with cpu, and auto-label when [eval_index](#traintoolcontrol_mainpy) meets the requirement [eval_threshold](#traintoolcontrol_mainpy), 3: Evaluate all models}
* pipeline_config: Name of config file.

#### train/input/4_setting/config
* num_classes: The number of object types to be detected.
* fixed_shape_resizer: The height and width of model you want.
* train_input_reader\tf_record_input_reader\input_path: Path to training tfrecord. '.record-?????-of-00010' should be added since sharded tfrecord are used. For example, "D:\\object_detection\\train\\input\\1_train_data\\tfrecord\\1_train_data.record-?????-of-00010". 
* eval_input_reader\tf_record_input_reader\input_path: Path to evaluating tfrecord. '.record-?????-of-00010' should be added.
* label_map_path: Path to pascal_label_map.pbtxt.


### Advanced Parameters Setting (Optional)
#### train/tool/control_main.py
* first_env["CUDA_VISIBLE_DEVICES"]: Main hardware setting used for training. '0' means use the first gpu, and multi-gpu can be used by setting the parameter to '0, 1, ...'
* second_env["CUDA_VISIBLE_DEVICES"]: Hardware setting for evaluating while training. '-1' means use cpu.
* eval_index: Evaluation metric used as monitoring indicator to start auto-labeling. More detail can be found on [COCO evaluation metrics](https://cocodataset.org/#detection-eval).
* eval_threshold: Requirement threshold of evaluation metric to start auto-labeling.
* IOU_threshold: The IOU threshold used to determine the confidence score threshold of each object during automatic labeling.

## Start Training
Run 'train/run.bat', models will be generated under 'train/train/'.

## Export Model
After finishing training, copy    
1. Config,  
2. The model wanted from [Start Training](#start-training)

to 'export_model/input/', then edit the model step in 'export_model/input/checkpoint'.  
Run 'export_model/export_inference_graph.bat', model will be exported to the output folder. (The model be used later is actually the folder 'export_model/output/saved_model')

## Inference
To check how well the object detection model is, only three steps are needed.  
Step 1. Copy pascal_label_map.pbtxt to 'inference/'.  
Step 2. Open 'inference/config.ini' and set some parameters
* img_path: Path to directory of image be detected.
* log_path: Path to predicted output folder.
* model_path: Path to model from [Export Model](#export-model).
* model_name: Model name.
* model_width: Model width while training.
* model_height: Model height while training.
* model_channel: Model channel while training, usually is 3 if [convert_to_grayscale](https://github.com/tensorflow/models/blob/master/research/object_detection/protos/image_resizer.proto) is not set to True during training.
* threshold: Threshold of confidence score that objects will be shown when box_num is set to None.
* box_num: The number of objects to be displayed on the image.
* gpu_memory_fraction: A value between 0 and 1 that indicates what fraction of the available GPU memory to pre-allocate for each process.

Step 3. Run 'inference/run.bat', result will be saved under log_path just specified.
