import math
import os
import tensorflow as tf
from tensorflow.python.keras.models import load_model
from tensorflow.python.keras.preprocessing.image import ImageDataGenerator
from tensorflow.python.keras.applications.imagenet_utils import preprocess_input
from tensorflow.python.keras.utils.vis_utils import plot_model
from tensorflow.python.keras.applications import xception, nasnet, mobilenet_v2, inception_v3
from tensorflow.python.keras.callbacks import CSVLogger, ModelCheckpoint, ReduceLROnPlateau, LearningRateScheduler, TensorBoard
import numpy as np
import configparser


def read_config():
    
    config = configparser.ConfigParser()
    config.read('config.ini')
    # assign gpu
    DEVICES = config.get('CUDA_VISIBLE_DEVICES', 'DEVICES')
    os.environ["CUDA_VISIBLE_DEVICES"] = DEVICES

    global TRAIN_IMG_PATH, CLASSES_NUM
    TRAIN_IMG_PATH = config.get('Image', 'train_img_path')
    CLASSES_NUM = config.getint('Image', 'classes_num')


    global PROJECT_NAME, MODEL_WIDTH, MODEL_HEIGHT, MODEL_CHANNEL, BATCH_SIZE, EPOCH, VALIDATION_SPLIT_RATE, INITIAL_EPOCH, SAVE_MODEL_TYPE, MODEL_TYPE, OPTIMIZER, LEARNING_RATE, MONITOR, LOSS_FUNCTION
    PROJECT_NAME = config.get('Model', 'project_name')
    MODEL_WIDTH = config.getint('Model', 'model_width')
    MODEL_HEIGHT = config.getint('Model', 'model_height')
    MODEL_CHANNEL = config.getint('Model', 'model_channel')
    BATCH_SIZE = config.getint('Model', 'batch_size')
    EPOCH = config.getint('Model', 'epoch')
    VALIDATION_SPLIT_RATE = config.getfloat('Model', 'validation_split_rate')
    INITIAL_EPOCH = config.getint('Model', 'initial_epoch')
    SAVE_MODEL_TYPE = config.get('Model', 'save_model_type')
    
    MODEL_TYPE = config.get('Model', 'model_type')
    OPTIMIZER = config.get('Model', 'optimizer')
    LEARNING_RATE = config.getfloat('Model', 'learning_rate')
    MONITOR = config.get('Model', 'monitor')
    LOSS_FUNCTION = config.get('Model', 'loss_function')
    
    
    
    global ROTATION_RANGE, VERTICAL_FLIP, HORIZONTAL_FLIP
    ROTATION_RANGE = config.getint('Data_Aug', 'rotation_range')
    VERTICAL_FLIP = config.get('Data_Aug', 'vertical_flip')
    HORIZONTAL_FLIP = config.get('Data_Aug', 'horizontal_flip')
    

def data_processing():
    
    if MODEL_CHANNEL == 1:
        keras_color_mode = "grayscale"
    else:
        keras_color_mode = "rgb"
    
    train_datagen = ImageDataGenerator(preprocessing_function=None,
                                       rescale=1./255,
                                       validation_split=VALIDATION_SPLIT_RATE,
                                       rotation_range=ROTATION_RANGE,
                                       zca_whitening=False,
                                       vertical_flip=VERTICAL_FLIP,
                                       horizontal_flip=HORIZONTAL_FLIP)
        
    train_batches = train_datagen.flow_from_directory(
        TRAIN_IMG_PATH,
        shuffle=True,
        target_size=(MODEL_HEIGHT, MODEL_WIDTH),
        color_mode=keras_color_mode,
        subset='training',
        batch_size=BATCH_SIZE,
        interpolation='nearest')



    valid_batches = train_datagen.flow_from_directory(
        TRAIN_IMG_PATH,
        shuffle=True,
        target_size=(MODEL_HEIGHT, MODEL_WIDTH),
        color_mode=keras_color_mode,
        subset='validation',
        batch_size=BATCH_SIZE,
        interpolation='nearest')

    # prevent validation steps being smaller than one 
    if valid_batches.samples / BATCH_SIZE < 1:
        validation_steps = 1
    else:
        validation_steps = math.ceil(valid_batches.samples / BATCH_SIZE)
        
    return train_batches, valid_batches, validation_steps



def model_setting():
    if not os.path.isdir('.\\h5'):
        os.makedirs('.\\h5')
    
    # whether load pre-trained model 
    if INITIAL_EPOCH != 0:
        model = tf.keras.models.load_model(
            '.\\h5\\'+'%s_%d.h5' % (PROJECT_NAME, INITIAL_EPOCH))
    else:       
        if MODEL_TYPE == 'DashNet':
            from model.dashnet import DashNet
            model = DashNet(input_shape=(MODEL_HEIGHT, MODEL_WIDTH, MODEL_CHANNEL), pooling='avg', classes=CLASSES_NUM)
        
        elif MODEL_TYPE == 'ShuffleNet':
            from model.shufflenet import ShuffleNet
            model = ShuffleNet(input_shape=(MODEL_HEIGHT, MODEL_WIDTH, MODEL_CHANNEL), pooling='avg', classes=CLASSES_NUM, groups=1)
            
        elif MODEL_TYPE == 'SqueezeNet':
            from model.squeezenet import SqueezeNet
            model = SqueezeNet(input_shape=(MODEL_HEIGHT, MODEL_WIDTH, MODEL_CHANNEL), pooling='avg', classes=CLASSES_NUM, weights=None, )

        elif MODEL_TYPE == 'Xception':
            model = xception.Xception(input_shape=(MODEL_HEIGHT, MODEL_WIDTH, MODEL_CHANNEL), pooling='avg', include_top=True, weights=None, input_tensor=None, classes=CLASSES_NUM)
                    
        elif MODEL_TYPE == 'NASNet':
            model = nasnet.NASNetMobile(input_shape=(MODEL_HEIGHT, MODEL_WIDTH, MODEL_CHANNEL), pooling='avg', include_top=True, weights=None, input_tensor=None, classes=CLASSES_NUM)

        elif MODEL_TYPE == 'Mobilenet':
            model = mobilenet_v2.MobileNetV2(input_shape=(MODEL_HEIGHT, MODEL_WIDTH, MODEL_CHANNEL), pooling='avg', include_top=True, weights=None, input_tensor=None, classes=CLASSES_NUM, alpha=1.0)
            
        elif MODEL_TYPE == 'InceptionV3':
            model = inception_v3.InceptionV3(input_shape=(MODEL_HEIGHT, MODEL_WIDTH, MODEL_CHANNEL), pooling='avg', include_top=True, weights=None, input_tensor=None, classes=CLASSES_NUM)
                
        else:
            print('Load Model Error!!! Please make sure you enter the correct model name!!!')
            
    return model

def save_model_setting():
    # 設定儲存模型的型態
    if SAVE_MODEL_TYPE == 'Best_acc':
        checkpoint = ModelCheckpoint(filepath='.\\h5\\'+'%s_{epoch:02d}.h5' % PROJECT_NAME, verbose=1, save_weights_only=False,
                                     monitor='val_acc', save_best_only=True, mode='auto')

    elif SAVE_MODEL_TYPE == 'Best_loss':
        checkpoint = ModelCheckpoint(filepath='.\\h5\\'+'%s_{epoch:02d}.h5' % PROJECT_NAME, verbose=1,save_weights_only=False,
                                     monitor='val_loss', save_best_only=True, mode='auto')

    elif SAVE_MODEL_TYPE == 'Always':
        checkpoint = ModelCheckpoint(filepath='.\\h5\\'+'%s_{epoch:02d}.h5' % PROJECT_NAME, verbose=1, save_weights_only=False)
    else:
        print('save_model_type error!!!')

    return checkpoint


def loss_func_setting():
    if LOSS_FUNCTION == 'categorical_crossentropy':
        loss_func = LOSS_FUNCTION
    
    elif LOSS_FUNCTION == 'focal_loss':
        from loss.focal_loss import category_focal_loss1
        loss_func = [category_focal_loss1(alpha=[2, 1], gamma=1)]
    return loss_func
    
def learning_rate_setting():
    if MODEL_WIDTH > 800:
        reduce_lr = ReduceLROnPlateau(monitor=MONITOR, patience=5, factor=0.5, verbose=1, mode='auto',
                                      cooldown=5,min_lr=0.00001)
    else:
        reduce_lr = ReduceLROnPlateau(monitor=MONITOR, patience=5, factor=0.7, verbose=1, mode='auto',
                                      cooldown=10, min_lr=0.00001)
    return reduce_lr
        
def optimizer_setting(model, loss_func):
    if OPTIMIZER == 'momentum':
        # decay=5e-4
        model.compile(optimizer=tf.keras.optimizers.SGD(learning_rate=LEARNING_RATE, momentum=0.9),
                      loss=loss_func,
                      metrics=['accuracy'])
        
    elif OPTIMIZER == 'NAG':
        model.compile(optimizer=tf.keras.optimizers.SGD(learning_rate=LEARNING_RATE, momentum=0.9, nesterov=True),
                      loss=loss_func,
                      metrics=['accuracy'])
        
    elif OPTIMIZER == 'Adam':
        model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=LEARNING_RATE, beta_1=0.9, beta_2=0.999),
                      loss=loss_func,
                      metrics=['accuracy'])
        
    elif OPTIMIZER == 'Nadam':
        model.compile(optimizer=tf.keras.optimizers.Nadam(learning_rate=LEARNING_RATE, beta_1=0.9, beta_2=0.999),
                      loss=loss_func,
                      metrics=['accuracy'])
    
    else:
        model.compile(optimizer=OPTIMIZER, 
                      loss=loss_func,
                      metrics=['accuracy'])

    return model
    
if __name__ == '__main__':
    # Step1 read config setting
    read_config()
    
    # Step2 load and split data, then do image pre-processing
    train_batches, valid_batches, validation_steps = data_processing()
    
    # check the order of class label 
    print('=============== order of class label ===============')
    for cls_, idx in train_batches.class_indices.items():
        print('Class: {} = {}'.format(idx, cls_))
    
    # Step3 set loss function
    loss_func = loss_func_setting()
    
    # Step4 set model and optimizer
    model = model_setting()
    model = optimizer_setting(model, loss_func)
    
    # Step5 set callbacks
    reduce_lr = learning_rate_setting()
    checkpoint = save_model_setting() 
    csv_logger = CSVLogger('%s_log.csv' % PROJECT_NAME, append=(INITIAL_EPOCH is not 0))
    tensorboard = TensorBoard(log_dir='.\\logs', histogram_freq=0, batch_size=BATCH_SIZE, write_graph=True,
                              write_grads=True, write_images=True, embeddings_freq=0, embeddings_layer_names=None, embeddings_metadata=None)
    callbacks = [reduce_lr, checkpoint, csv_logger, tensorboard]
    
    # Step6 model training
    model.fit_generator(
        train_batches,
        steps_per_epoch = train_batches.samples / BATCH_SIZE,
        validation_data = valid_batches,
        validation_steps = validation_steps,
        epochs = EPOCH,
        shuffle = True,
        workers = 7,
        initial_epoch=INITIAL_EPOCH,
        use_multiprocessing = False,
        callbacks = callbacks)
