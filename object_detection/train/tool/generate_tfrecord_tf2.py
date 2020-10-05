"""
Usage:
  # From tensorflow/models/
  # Create train data:
  python generate_tfrecord.py --csv_input=images/train_labels.csv --image_dir=images/train --output_path=train.record

  # Create test data:
  python generate_tfrecord.py --csv_input=images/test_labels.csv  --image_dir=images/test --output_path=test.record
"""
from __future__ import division
from __future__ import print_function
from __future__ import absolute_import

import contextlib2
import os
import io
import pandas as pd
import tensorflow as tf
import re

from PIL import Image
from object_detection.utils import dataset_util
from object_detection.dataset_tools import tf_record_creation_util
from collections import namedtuple, OrderedDict
from absl import flags


flags.DEFINE_string('input_dir', '', 'Path to the input folder')
flags.DEFINE_string('sharding', '1', 'Whether shard tfrecord')
FLAGS = flags.FLAGS


def parse_pbtxt(pbtxt_path):
    my_dict = {}
    with open(pbtxt_path, 'r') as f:
        key_ = None; value_ = None
        for line in f:
            if '}' in line:
                my_dict[key_] = int(value_)
                key_ = None
                value_ = None
                search = False
            if 'item' in line:
                search = True
            if search and ('name' in line):
                key_ = line.split(':')[-1]
                key_ = re.search(r'[a-zA-Z0-9_-]+', key_).group(0)

            if search and ('id' in line):
                value_ = line.split(':')[-1]
                value_ = re.search(r'[0-9]+', value_).group(0)
    return my_dict

def split(df, group):
    data = namedtuple('data', ['filename', 'object'])
    gb = df.groupby(group)
    return [data(filename, gb.get_group(x)) for filename, x in zip(gb.groups.keys(), gb.groups)]


def create_tf_example(group, path, my_dict):
    with tf.io.gfile.GFile(os.path.join(path, '{}'.format(group.filename)), 'rb') as fid:
        encoded_jpg = fid.read()
    encoded_jpg_io = io.BytesIO(encoded_jpg)
    image = Image.open(encoded_jpg_io)
    width, height = image.size

    filename = group.filename.encode('utf8')
    image_format = b'jpg'
    xmins = []
    xmaxs = []
    ymins = []
    ymaxs = []
    classes_text = []
    classes = []

    for index, row in group.object.iterrows():
        xmins.append(row['xmin'] / width)
        xmaxs.append(row['xmax'] / width)
        ymins.append(row['ymin'] / height)
        ymaxs.append(row['ymax'] / height)
        classes_text.append(row['class'].encode('utf8'))
        classes.append(my_dict[row['class']])


    tf_example = tf.train.Example(features=tf.train.Features(feature={
        'image/height': dataset_util.int64_feature(height),
        'image/width': dataset_util.int64_feature(width),
        'image/filename': dataset_util.bytes_feature(filename),
        'image/source_id': dataset_util.bytes_feature(filename),
        'image/encoded': dataset_util.bytes_feature(encoded_jpg),
        'image/format': dataset_util.bytes_feature(image_format),
        'image/object/bbox/xmin': dataset_util.float_list_feature(xmins),
        'image/object/bbox/xmax': dataset_util.float_list_feature(xmaxs),
        'image/object/bbox/ymin': dataset_util.float_list_feature(ymins),
        'image/object/bbox/ymax': dataset_util.float_list_feature(ymaxs),
        'image/object/class/text': dataset_util.bytes_list_feature(classes_text),
        'image/object/class/label': dataset_util.int64_list_feature(classes),
    }))
    return tf_example


def main(_):
    for folder in ['1_train_data', '2_eval_data']:
        # set the path
        csv_input = os.path.join(FLAGS.input_dir, folder, 'tfrecord', folder+'.csv')
        image_dir = os.path.join(FLAGS.input_dir, folder, 'img')
        output_path = os.path.join(FLAGS.input_dir, folder, 'tfrecord', folder+'.record')
        pbtxt_path = os.path.join(FLAGS.input_dir, '4_setting/pascal_label_map.pbtxt')
        
        num_shards=10
        path = os.path.join(os.getcwd(), image_dir)
        examples = pd.read_csv(csv_input)
        grouped = split(examples, 'filename')
        my_dict = parse_pbtxt(pbtxt_path)
        
        if FLAGS.sharding == '1':
            with contextlib2.ExitStack() as tf_record_close_stack:
                output_tfrecords = tf_record_creation_util.open_sharded_output_tfrecords(
                    tf_record_close_stack, output_path, num_shards)
                count = 0
                for group in grouped:
                    tf_example = create_tf_example(group, path, my_dict)
                    output_shard_index = count % num_shards
                    output_tfrecords[output_shard_index].write(tf_example.SerializeToString())
                    count += 1
                
        else:
            writer = tf.io.TFRecordWriter(output_path)
            for group in grouped:
                tf_example = create_tf_example(group, path, my_dict)
                writer.write(tf_example.SerializeToString())
            writer.close()
        
    print('Successfully create the TFRecords')



if __name__ == '__main__':
    tf.compat.v1.app.run()
