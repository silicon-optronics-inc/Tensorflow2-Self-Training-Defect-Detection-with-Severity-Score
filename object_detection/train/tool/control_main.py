# -*- coding: utf-8 -*-
"""
Created on Wed Sep 16 10:54:30 2020

@author: martinchen
"""

from absl import app, flags
import os
import subprocess
import sys
import time

flags.DEFINE_string('mode', '', 'Mode of program,' 
                    '0: train only,'
                    '1: train + evaluate,'
                    '2: semi-supervised learning,'
                    '3: evaluate only')
flags.DEFINE_string('work_directory', '', 'Work directory')
flags.DEFINE_string('pipeline_config', '', 'Name of pipeline config used')
FLAGS = flags.FLAGS

def setting():
    input_dir = os.path.join(FLAGS.work_directory, 'input')
    model_dir = os.path.join(FLAGS.work_directory, 'train')
    pipeline_config_path = os.path.join(input_dir, '4_setting', FLAGS.pipeline_config)
    gpu_env = os.environ.copy()
    cpu_env = os.environ.copy()
    gpu_env["CUDA_VISIBLE_DEVICES"] = '0'
    cpu_env["CUDA_VISIBLE_DEVICES"] = '-1'
    
    # Evaluation index and threshold to ensure quality labels. Start auto-label after reaching the requirement threshold.
    eval_index = 'DetectionBoxes_Precision/mAP@.50IOU'
    eval_threshold = 0.8
    # The IOU threshold used to determine the confidence score threshold of each object during automatic labeling
    IOU_threshold = 0.7
    
    return input_dir, model_dir, pipeline_config_path, gpu_env, cpu_env, eval_index, eval_threshold, IOU_threshold

def main(argv):
    input_dir, model_dir, pipeline_config_path, gpu_env, cpu_env, eval_index, eval_threshold, IOU_threshold = setting()
    
    if FLAGS.mode == '0':
        py_xml2csv = subprocess.Popen(['python', 'tool/xml_to_csv.py', input_dir], shell=False)
        py_xml2csv.wait()
        py_generate_tfrecord = subprocess.Popen(['python', 'tool/generate_tfrecord_tf2.py', '--input_dir='+input_dir, '--sharding=1'], shell=False)
        py_generate_tfrecord.wait()
        py_train = subprocess.Popen(['python', 'tool/model_main_tf2.py', '--model_dir='+model_dir, '--pipeline_config_path='+pipeline_config_path,'--logalsotostderr'], env=gpu_env, shell=False)
        py_train.wait()
        
    elif FLAGS.mode == '1':
        py_xml2csv = subprocess.Popen(['python', 'tool/xml_to_csv.py', input_dir], shell=False)
        py_xml2csv.wait()
        py_generate_tfrecord = subprocess.Popen(['python', 'tool/generate_tfrecord_tf2.py', '--input_dir='+input_dir, '--sharding=1'], shell=False)
        py_generate_tfrecord.wait()
        py_train = subprocess.Popen(['python', 'tool/model_main_tf2.py', '--model_dir='+model_dir, '--pipeline_config_path='+pipeline_config_path,'--logalsotostderr'], env=gpu_env, shell=False)
        py_eval = subprocess.Popen(['python', 'tool/eval.py', '--model_dir='+model_dir, '--pipeline_config_path='+pipeline_config_path, '--mode='+FLAGS.mode], env=cpu_env, creationflags=subprocess.CREATE_NEW_CONSOLE, shell=False)
        py_eval.wait()
        
    elif FLAGS.mode == '2':
        while True:
            py_xml2csv = subprocess.Popen(['python', 'tool/xml_to_csv.py', input_dir], shell=False)
            py_xml2csv.wait()
            py_generate_tfrecord = subprocess.Popen(['python', 'tool/generate_tfrecord_tf2.py', '--input_dir='+input_dir, '--sharding=1'], shell=False)
            py_generate_tfrecord.wait()
            py_train = subprocess.Popen(['python', 'tool/model_main_tf2.py', '--model_dir='+model_dir, '--pipeline_config_path='+pipeline_config_path,'--logalsotostderr'], env=gpu_env, shell=False)
            py_eval = subprocess.Popen(['python', 'tool/eval.py', '--model_dir='+model_dir, '--pipeline_config_path='+pipeline_config_path, '--mode='+FLAGS.mode, '--eval_index'+eval_index, '--eval_threshold'+str(eval_threshold)], env=cpu_env, creationflags=subprocess.CREATE_NEW_CONSOLE, shell=False)
            py_eval.wait()
            py_train.terminate()
            py_auto_label = subprocess.Popen(['python', 'tool/auto_label.py', '--model_dir='+model_dir, '--pipeline_config_path='+pipeline_config_path, '--input_dir='+input_dir, '--IOU_threshold'+str(IOU_threshold)], env=gpu_env, shell=False)
            py_auto_label.wait()
        
    elif FLAGS.mode == '3':
        py_eval = subprocess.Popen(['python', 'tool/eval.py', '--model_dir='+model_dir, '--pipeline_config_path='+pipeline_config_path, '--mode='+FLAGS.mode], env=gpu_env, shell=False)
        py_eval.wait()
        print('Evaluation done!')
        
    else:
        print('Wrong mode number!!!')
        sys.exit()
        
    
    

    
if __name__ == '__main__':
    app.run(main)