# Classification Tool Guide
This document will go through how to train the classification model and use it for inference.

## Data Preparation
Firstly, one must assign different categories of pictures to different folders. That is, one can create an folder name 'data/' and create N folders under it corresponding to the category name if there are N categories to be classified.
To train a model that can give a severity score, it is recommended to name the folder from 1 to N, and move the images to the corresponding folders in order from light to heavy.

## Parameters
Open 'train/config.ini'
### Basic Parameters Setting
* train_img_path: Path to train images from [Data Preparation](#data-preparation).
* classes_num: Number of categories to be classified.
* project_name: Project name.
* model_width: Model width.
* model_height: Model height.
* model_channel: Model channel, should be set to 3 if one use RGB images, 1 for grayscale.
* batch_size: Batch size while training.
* model_type: Model structrue to be trained.
* initial_epoch: Step of pretrained model to load and keep training. 

### Advanced Parameters Setting (Optional)
* save_model_type: When to save model.
* epoch: Number of epoch to stop training.
* validation_split_rate: Proportion of data divided into validation set.
* optimizer: Optimizer be used.
* learning_rate: Learning rate.
* monitor: Evaluation metric be used. 
* loss_function: Loss function, one can define own loss function under 'train/loss/' just as focal_loss.py this project used.
* rotation_range: rotation_range used in [keras.preprocessing.image.ImageDataGenerator](https://keras.io/api/preprocessing/image/).
* vertical_flip: vertical_flip used in [keras.preprocessing.image.ImageDataGenerator](https://keras.io/api/preprocessing/image/).
* horizontal_flip: horizontal_flip used in [keras.preprocessing.image.ImageDataGenerator](https://keras.io/api/preprocessing/image/).
* DEVICES: Hardware setting used for training. '0' means use the first gpu, and multi-gpu can be used by setting the parameter to '0, 1, ...'

## Start Training
Run 'train/train.bat', models will be generated under 'train/h5/'.

## Inference
To check how well the classification model is, only two steps are needed.  
Step 1. Open 'inference/config.ini' and set some parameters
* img_path: Path to directory of images be classified.
* classes: Name of each categories, seperated by comma. 
* log_path: Path to predicted output folder.
* model_path: Path to model from [Start Training](#start-training).
* model_name: Model name.
* model_width: Model width while training.
* model_height: Model height while training.
* model_channel: Model channel while training.

Step 2. run 'inference.bat', result will be saved under log_path just specified.
