import configparser
import cv2
import csv
import datetime
import numpy as np
import os
import re
import tensorflow as tf 
import time


def show_time_taken(tStart):
    t_hour, temp_sec = divmod(time.time() - tStart, 3600)
    t_min, t_sec = divmod(temp_sec, 60)
    msg = '{} hours, {} mins, {} seconds'.format(int(t_hour), int(t_min), int(t_sec))
    return msg

def read_ini():
    config = configparser.ConfigParser()
    config.read('config.ini') 
    
    global DATE
    DATE = datetime.date.today().strftime('%Y%m%d')
    
    # image
    global IMG_PATH
    IMG_PATH = config.get('Image', 'img_path')

    # model
    global MODEL_PATH, OBJ_MODEL_NAME, OBJ_MODEL_WIDTH, OBJ_MODEL_HEIGHT, OBJ_MODEL_CHANNEL, CLS_MODEL_NAME, CLS_MODEL_WIDTH, CLS_MODEL_HEIGHT, CLS_MODEL_CHANNEL
    MODEL_PATH = config.get('Model', 'model_path')
    OBJ_MODEL_NAME = config.get('Model','obj_model_name')
    OBJ_MODEL_WIDTH = config.getint('Model', 'obj_model_width')
    OBJ_MODEL_HEIGHT = config.getint('Model', 'obj_model_height')
    OBJ_MODEL_CHANNEL = config.getint('Model','obj_model_channel')
    CLS_MODEL_NAME = config.get('Model','cls_model_name')
    CLS_MODEL_WIDTH = config.getint('Model', 'cls_model_width')
    CLS_MODEL_HEIGHT = config.getint('Model', 'cls_model_height')
    CLS_MODEL_CHANNEL = config.getint('Model','cls_model_channel')
    
    
    # output
    global LOG_PATH
    LOG_PATH = config.get('Log', 'log_path')
    LOG_PATH = os.path.join(LOG_PATH, '{}_{}_{}'.format(DATE, OBJ_MODEL_NAME.split('.')[0], CLS_MODEL_NAME.split('.')[0]))

    
    # other
    global THRESHOLD, BOX_NUM, GPU_MEMORY_FRACTION
    THRESHOLD = config.getfloat('Other', 'threshold')
    BOX_NUM = config.get('Other', 'box_num')
    GPU_MEMORY_FRACTION = config.getfloat('Other','gpu_memory_fraction')
    
    
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

def load_obj_model():
    print('loading object detection model...')
    return tf.saved_model.load(os.path.join(MODEL_PATH, OBJ_MODEL_NAME))

def load_cls_model():
    print('loading classification model...')
    return tf.keras.models.load_model(os.path.join(MODEL_PATH, CLS_MODEL_NAME), compile=False)


def predict_obj_img(obj_model, img):
    input_tensor = tf.convert_to_tensor(img)
    input_tensor = input_tensor[tf.newaxis,...]
    detections = obj_model(input_tensor)
    num_detections = int(detections.pop('num_detections'))
    detections = {key:value[0, :num_detections].numpy() 
                 for key,value in detections.items()}
    
    return detections['detection_boxes'], detections['detection_scores'], detections['detection_classes'], num_detections


def predict_cls_img(cls_model, img, obj_boxes, obj_classes, show_box_num):
    def adjust_boundary(crop_min, crop_max):
        # prevent coordinate from being out of boundary 
        if crop_min < 0:
            crop_max = crop_max - crop_min
            crop_min = 0
        if crop_max > OBJ_MODEL_WIDTH:
            crop_min = crop_min - (crop_max - OBJ_MODEL_WIDTH)
            crop_max = OBJ_MODEL_WIDTH
        return crop_min, crop_max
    
    def cls_img_preprocess(img):
        img = cv2.resize(img, (CLS_MODEL_WIDTH, CLS_MODEL_HEIGHT), interpolation = cv2.INTER_LINEAR)
        if CLS_MODEL_CHANNEL == 1 and OBJ_MODEL_CHANNEL == 3:
            img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
            img = np.expand_dims(img, axis=2)
    
        img = np.array(img) / 255
        img = np.expand_dims(img, axis=0)
        return img
    
    
    cls_infor = []
    # prevent obj_boxes being empty
    for i in range(min(len(obj_boxes), show_box_num)):
        ymin, xmin, ymax, xmax = obj_boxes[i]
        ymin = int(ymin * OBJ_MODEL_HEIGHT)
        xmin = int(xmin * OBJ_MODEL_WIDTH)
        ymax = int(ymax * OBJ_MODEL_HEIGHT)
        xmax = int(xmax * OBJ_MODEL_WIDTH)
        
        # crop image for classification
        center = int((xmin+xmax) / 2)
        crop_min, crop_max = adjust_boundary(center - 9, center + 9)
        crop_img = img[:, crop_min:crop_max]
        crop_img = cls_img_preprocess(crop_img)

        scores = cls_model.predict(crop_img)[0]
        score = 0
        for j in range(len(scores)):
            score += scores[j]*0.2*(j)
        score = str('%.4f'%score)
        cls_infor.append([obj_classes[i], score, ymin, xmin, ymax, xmax])
        
    return cls_infor



def process_log(file, img, cls_infor, pbtxt_dict):
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
            

    # generate colour table
    colour = [[0,0,205], [238,102,0],[143,188,143],[139,28,98],[105,105,105],[70,130,180]]
    if show_box_num > len(colour):
        np.random.seed(3600)
        colour_ext = np.random.randint(50, 256, size=(show_box_num - len(colour), 3))
        colour = colour + colour_ext.tolist()
    
    # default log message if no defect detected
    log_msg = [file, show_box_num]
    
    # draw image and write log message
    for i in range(len(cls_infor)):
        classid = int(cls_infor[i][0])
        img = draw_bndbox(img, classid, cls_infor[i][1], 0, cls_infor[i][3], OBJ_MODEL_HEIGHT, cls_infor[i][5], pbtxt_dict, colour, i)
        log_msg = log_msg + [classid, cls_infor[i][1], cls_infor[i][2], cls_infor[i][3], cls_infor[i][4], cls_infor[i][5]]

    # write drawn image
    img_draw_folder = LOG_PATH + '\\draw'
    if not os.path.exists(img_draw_folder):
        os.makedirs(img_draw_folder)
    cv2.imwrite(img_draw_folder + '\\' + file, img)

    # record log
    record_log(log_msg)
    
if __name__ == '__main__':
    total_start = time.time()
    # Step1 read config
    read_ini()
    # Step2 parse pbtxt
    pbtxt_dict = parse_pbtxt()
    
    # Step3 preload object detection model
    obj_model = load_obj_model()
    
    # Step4 preload classification model
    cls_model = load_cls_model()
    
    # iteratelly inference 
    for root, dirs, files in os.walk(IMG_PATH):
        for file in files:
            print(file)
            t_start = time.time()
            
            # read image
            cv2_channel = 0 if OBJ_MODEL_CHANNEL == 1 else 1
            img = cv2.imread(os.path.join(root, file), cv2_channel)
            
            if OBJ_MODEL_CHANNEL == 3:
                img = cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
            
            # object detection
            img = cv2.resize(img, (OBJ_MODEL_WIDTH, OBJ_MODEL_HEIGHT), interpolation = cv2.INTER_LINEAR)
            obj_boxes, obj_scores, obj_classes, obj_num_detections = predict_obj_img(obj_model, img)
            
            # get the number of defects that scores is greater than threshold
            show_box_num = int(BOX_NUM) if BOX_NUM != 'None' else len(obj_scores[obj_scores >= THRESHOLD])
            
            # classification
            cls_infor = predict_cls_img(cls_model, img, obj_boxes, obj_classes, show_box_num)
            
            # record log
            process_log(file, img, cls_infor, pbtxt_dict)
            
            t_end = time.time()
            print('Takes {} second'.format(round(t_end - t_start, 2)))
            print('-------------------\n')
            
    print('Total takes {}'.format(show_time_taken(total_start)))
