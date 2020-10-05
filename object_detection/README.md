# Object Detection Training Guide
This document will explain how to use these programs to train the model and the parameters that need to be adjusted.

## Data Preparation
Four types of fils are needed. First two could be get if follow the instructions of [LabelImg github](https://github.com/tzutalin/labelImg). The last two files could be found under 'object_detection/train/input/4_setting/'.
1. Images
2. XML, follow PASCAL VOC format.
3. pascal_label_map.pbtxt, contains the id and name of the object to be detected. THe name and number of items should be identical to those in xml.
4. Config, configure the object detection training pipeline. One can define their own training process by modifying or replacing it with any configuration file in other online resources.

After preparing the required files above, copy images and XML to corresponding folder(img and xml) under 'object_detection/train/input/1_train_data/' and 'object_detection/train/input/2_eval_data/' respectively as training and evaluation data. 
To enable self-training, unlabeled images should be put under 'object_detection/train/input/3_unlabeled_data/img/'


## Parameters Setting
### object_detection/train/run.bat
* mode, 
* pipeline_config, name of config file.

### object_detection/train/input/4_setting/config
