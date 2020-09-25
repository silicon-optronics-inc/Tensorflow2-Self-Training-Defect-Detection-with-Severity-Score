# -*- coding: utf-8 -*-
"""
Created on Mon Jul  6 11:01:11 2020

@author: martinchen
"""

import configparser
import cv2
import csv
import datetime
import tensorflow.keras as keras
import time
import numpy as np
import os

def show_time_taken(tStart):
    t_hour, temp_sec = divmod(time.time() - tStart, 3600)
    t_min, t_sec = divmod(temp_sec, 60)
    msg = '{} hours, {} mins, {} seconds'.format(int(t_hour), int(t_min), int(t_sec))
    return msg

def read_config():
    print('reading config...')
    ini_file = 'config.ini'
    config = configparser.ConfigParser()
    config.read(ini_file)

    global DATE
    DATE = datetime.date.today().strftime('%Y%m%d')

    global IMAGE_PATH, CLASSES
    IMAGE_PATH = config.get('Image', 'image_path')
    CLASSES = config.get('Image', 'classes')
    CLASSES = [i.strip() for i in CLASSES.split(',')]
    

    global MODEL_WIDTH, MODEL_HEIGHT, MODEL_CHANNEL, MODEL_PATH, MODEL_NAME
    MODEL_WIDTH = config.getint('Model', 'model_width')
    MODEL_HEIGHT = config.getint('Model', 'model_height')
    MODEL_CHANNEL = config.getint('Model', 'model_channel')
    MODEL_PATH = config.get('Model', 'model_path')
    MODEL_NAME = config.get('Model', 'model_name')
    
    global LOG_PATH
    LOG_PATH = config.get('Log', 'log_path')
    LOG_PATH = os.path.join(LOG_PATH, MODEL_NAME.split('.')[0] + '_' + DATE)
    if not os.path.exists(LOG_PATH):
        os.makedirs(LOG_PATH)
    
    
    global THRESHOLD
    THRESHOLD = config.getfloat('Other_Setting', 'threshold')


    global EXT
    EXT = ['.png', '.PNG', '.jpg', '.JPG']


def load_model():
    print('loading model...')
    model = keras.models.load_model(os.path.join(MODEL_PATH, MODEL_NAME))

    return model


def inference(model):
    def img_preprocess(img):
        img = cv2.resize(img, (MODEL_WIDTH, MODEL_HEIGHT), interpolation = cv2.INTER_NEAREST)
        
        if MODEL_CHANNEL == 3:
            img = cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
            
        # when load image in grayscale, dimension of input would be 2D instead of 3D
        else:
            img = np.expand_dims(img, axis=2)
            
        img = np.array(img) / 255
        image_np_expanded = np.expand_dims(img, axis=0)
        return image_np_expanded
            
    def record_log(log_msg):
        if not os.path.isfile(LOG_PATH + '\\log.csv'):
            with open(LOG_PATH + '\\log.csv', 'w',newline='') as csv_file:
                writer = csv.writer(csv_file)
                writer.writerow(['FileName'] + CLASSES)
                csv_file.flush()
        with open(LOG_PATH + '\\log.csv', 'a',newline='') as csv_file:
            writer = csv.writer(csv_file)
            writer.writerow(log_msg)
            csv_file.flush()
        
    cv2_channel = 0 if MODEL_CHANNEL == 1 else 1
    for root, dirs, files in os.walk(IMAGE_PATH):
        for file in files:
            if file.endswith(tuple(EXT)):
                print(file)
                img = cv2.imread(os.path.join(root, file), cv2_channel)
                img = img_preprocess(img)
                scores = model.predict(img)[0]
                # print(scores)
                # if scores[0] > THRESHOLD:
                    # print(file, CLASSES[0])
                # else:
                    # print(file, CLASSES[1])
                # log_msg = [file]
                
                # print message
                top_inds = scores.argsort()[::-1]
                for i in top_inds:
                    print('    {:.3f}  {}'.format(scores[i], CLASSES[i]))
                print('--------------------')
                
                # record log
                scores = scores.tolist()
                log_msg = [file] + scores
                record_log(log_msg)


if __name__ == '__main__':
    total_start = time.time()
    read_config()
    model = load_model()
    inference(model)
    
    print('Total takes {}'.format(show_time_taken(total_start)))