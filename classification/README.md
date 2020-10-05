# Classification Tool Guide
This document will go through how to train the classification model and use it for inference.

## Data Preparation
Firstly, one must assign different categories of pictures to different folders. That is, one can create an folder name 'data/' and create 6 folders corresponding to the category name under it if there are 6 categories to be classified.

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
