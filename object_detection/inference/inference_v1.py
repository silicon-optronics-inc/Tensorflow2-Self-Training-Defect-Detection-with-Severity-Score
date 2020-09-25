import configparser
import cv2
import csv
import datetime
import numpy as np
import os
import re
import tensorflow as tf 
import time

def read_ini():
    config = configparser.ConfigParser()
    config.read('config.ini') 
    
    global DATE
    DATE = datetime.date.today().strftime('%Y%m%d')
    
    # image
    global IMG_PATH
    IMG_PATH = config.get('Image', 'img_path')
    
    # model
    global MODEL_PATH, MODEL_NAME, MODEL_WIDTH, MODEL_HEIGHT, MODEL_CHANNEL 
    MODEL_PATH = config.get('Model', 'model_path')
    MODEL_NAME = config.get('Model','model_name')
    MODEL_WIDTH = config.getint('Model', 'model_width')
    MODEL_HEIGHT = config.getint('Model', 'model_height')
    MODEL_CHANNEL = config.getint('Model','model_channel')
    
    # output
    global LOG_PATH
    LOG_PATH = config.get('Log', 'log_path')
    LOG_PATH = os.path.join(LOG_PATH, DATE + '_' + MODEL_NAME.split('.')[0])
    
    # other
    global THRESHOLD, BOX_NUM, GPU_MEMORY_FRACTION
    THRESHOLD = config.getfloat('Other', 'threshold')
    BOX_NUM = config.get('Other', 'box_num')
    GPU_MEMORY_FRACTION = config.getfloat('Other','gpu_memory_fraction')
    
def load_model():
    print('\nLoad Model...')
    return tf.saved_model.load(os.path.join(MODEL_PATH, MODEL_NAME))

def predict_img(img, obj_model):
    input_tensor = tf.convert_to_tensor(img)
    input_tensor = input_tensor[tf.newaxis,...]
    detections = obj_model(input_tensor)
    num_detections = int(detections.pop('num_detections'))
    detections = {key:value[0, :num_detections].numpy() 
                 for key,value in detections.items()}
    
    return detections['detection_boxes'], detections['detection_scores'], detections['detection_classes'], num_detections

def log_process(file, img, boxes, scores, classes, show_box_num):
    def parse_pbtxt():
        for file in os.listdir(os.getcwd()):
            if file.endswith('.pbtxt'):
                pbtxt_path = file
        pbtxt_dict = {}
        with open(pbtxt_path, 'r') as f:
            key_ = None; value_ = None
            for line in f:
                if '}' in line:
                    pbtxt_dict[int(key_)] = value_
                    key_ = None
                    value_ = None
                    search = False
                if 'item' in line:
                    search = True
                if search and ('name' in line):
                    value_ = line.split(':')[-1]
                    value_ = re.search(r'[a-zA-Z0-9_-]+', value_).group(0)
    
                if search and ('id' in line):
                    key_ = line.split(':')[-1]
                    key_ = re.search(r'[0-9]+', key_).group(0)
        return pbtxt_dict
    
    def draw_bndbox(img, classid, score, ymin, xmin, ymax, xmax, pbtxt_dict, colour, i):
        text_position_x = 20
        text_position_y = (1+i) * 20
        cv2.rectangle(img, (xmin,ymin), (xmax,ymax), tuple(colour[i]))
        msg = 'type:' + pbtxt_dict[classid] + ', score:' + score
        cv2.putText(img, msg, (text_position_x, text_position_y), cv2.FONT_HERSHEY_COMPLEX_SMALL, 0.8, tuple(colour[i]), 1)
        return img
        
    def record_log(log_msg):
        if not os.path.isfile(LOG_PATH + '\\log.csv'):
            with open(LOG_PATH + '\\log.csv', 'w',newline='') as csv_file:
                writer = csv.writer(csv_file)
                writer.writerow(['FileName', 'BoxNum', 'Class', 'Score', 'ymin', 'xmin', 'ymax', 'xmax'])
                csv_file.flush()

        with open(LOG_PATH + '\\log.csv', 'a',newline='') as csv_file:
            writer = csv.writer(csv_file)
            writer.writerow(log_msg)
            csv_file.flush()
    
    pbtxt_dict = parse_pbtxt()
    # gnenrate colour table
    colour = [[0,0,205], [238,102,0],[143,188,143],[139,28,98],[105,105,105],[70,130,180]]
    if show_box_num > len(colour):
        np.random.seed(3600)
        colour_ext = np.random.randint(50, 256, size=(show_box_num - len(colour), 3))
        colour = colour + colour_ext.tolist()
    log_msg = [file, show_box_num]
    
    # draw image and write log message
    img = cv2.resize(img, (MODEL_WIDTH, MODEL_HEIGHT), interpolation = cv2.INTER_NEAREST)
    for i in range(show_box_num):
        classid = int(classes[i])
        score = str('%.4f'%scores[i])
        ymin, xmin, ymax, xmax = boxes[i]
        ymin = int(ymin * MODEL_HEIGHT)
        xmin = int(xmin * MODEL_WIDTH)
        ymax = int(ymax * MODEL_HEIGHT)
        xmax = int(xmax * MODEL_WIDTH)
        
        img = draw_bndbox(img, classid, score, ymin, xmin, ymax, xmax, pbtxt_dict, colour, i)
        log_msg = log_msg + [pbtxt_dict[classid], score, ymin, xmin, ymax, xmax]

    # write drawn image
    img_draw_folder = LOG_PATH + '\\draw'
    if not os.path.exists(img_draw_folder):
        os.makedirs(img_draw_folder)
    cv2.imwrite(img_draw_folder + '\\' + file, img)

    # record log
    record_log(log_msg)
    
if __name__ == '__main__':
    total_start = time.time()
    read_ini()
    obj_model = load_model()
    
    for root, dirs, files in os.walk(IMG_PATH):
        for file in files:
            t_start = time.time()
            cv2_channel = 0 if MODEL_CHANNEL == 1 else 1
            img = cv2.imread(os.path.join(root, file), cv2_channel)
            if MODEL_CHANNEL == 3:
                img = cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
            
            result_boxes, result_scores, result_classes, result_num_detections = predict_img(img, obj_model)
            # get the number of defects with a scores greater than the threshold
            if BOX_NUM != 'None':
                show_box_num = int(BOX_NUM)
            else:
                show_box_num = len(result_scores[result_scores >= THRESHOLD])
            print(file)

            log_process(file, img, result_boxes, result_scores, result_classes, show_box_num)
            t_end = time.time()
            print('Takes {} second'.format(round(t_end - t_start, 2)))
            print('-------------------\n')
    total_end = time.time()
    print('Takes {} second'.format(round(total_end - total_start, 2)))