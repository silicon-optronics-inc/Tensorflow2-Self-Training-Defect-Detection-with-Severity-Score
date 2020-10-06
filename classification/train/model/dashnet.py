# -*- coding: utf-8 -*-
"""
Created on Wed Jul  8 10:31:22 2020

@author: martinchen
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os

from tensorflow.python.keras import backend
from tensorflow.python.keras import layers
from tensorflow.python.keras.applications import imagenet_utils
from tensorflow.python.keras.engine import training
from tensorflow.python.keras.utils import data_utils
from tensorflow.python.keras.utils import layer_utils
from tensorflow.python.util.tf_export import keras_export



def conv2d_bn(x,
              filters,
              num_row,
              num_col,
              padding='same',
              strides=(1, 1),
              name=None):
  """Utility function to apply conv + BN.

  Arguments:
    x: input tensor.
    filters: filters in `Conv2D`.
    num_row: height of the convolution kernel.
    num_col: width of the convolution kernel.
    padding: padding mode in `Conv2D`.
    strides: strides in `Conv2D`.
    name: name of the ops; will become `name + '_conv'`
      for the convolution and `name + '_bn'` for the
      batch norm layer.

  Returns:
    Output tensor after applying `Conv2D` and `BatchNormalization`.
  """
  if name is not None:
    bn_name = name + '_bn'
    conv_name = name + '_conv'
  else:
    bn_name = None
    conv_name = None
  if backend.image_data_format() == 'channels_first':
    bn_axis = 1
  else:
    bn_axis = 3
  x = layers.Conv2D(
      filters, (num_row, num_col),
      strides=strides,
      padding=padding,
      use_bias=False,
      name=conv_name)(
          x)
  x = layers.BatchNormalization(axis=bn_axis, scale=False, name=bn_name)(x)
  x = layers.Activation('relu', name=name)(x)
  return x


def DashNet(input_tensor=None, input_shape=None, pooling='avg', classes=2, classifier_activation='softmax'):
    if input_tensor is None:
        img_input = layers.Input(shape=input_shape)
    else:
        if not backend.is_keras_tensor(input_tensor):
            img_input = layers.Input(tensor=input_tensor, shape=input_shape)
        else:
            img_input = input_tensor
    
    if backend.image_data_format() == 'channels_first':
        channel_axis = 1
    else:
        channel_axis = 3
    
    # customize model structure
    x = conv2d_bn(img_input, 32, 15, 3, strides=(3, 1), padding='valid')
    x = conv2d_bn(x, 32, 15, 3, padding='same')
    x = conv2d_bn(x, 64, 15, 3, padding='same')
    x = layers.MaxPooling2D(pool_size=(15, 3), strides=(5, 1))(x)

    x = conv2d_bn(x, 80, 1, 1, padding='same')
    x = conv2d_bn(x, 192, 15, 3, padding='same')
    x = layers.MaxPooling2D(pool_size=(15, 3), strides=(5, 2))(x)

    # mixed 0:
    branch1x1 = conv2d_bn(x, 64, 1, 1)

    branch5x5 = conv2d_bn(x, 48, 1, 1)
    branch5x5 = conv2d_bn(branch5x5, 64, 5, 5)

    branch3x3dbl = conv2d_bn(x, 64, 1, 1)
    branch3x3dbl = conv2d_bn(branch3x3dbl, 96, 3, 3)
    branch3x3dbl = conv2d_bn(branch3x3dbl, 96, 3, 3)

    branch_avg_pool = layers.AveragePooling2D((3, 3), strides=(1, 1), padding='same')(x)
    branch_avg_pool = conv2d_bn(branch_avg_pool, 32, 1, 1)
    
    
    
    x = layers.concatenate([branch1x1, branch5x5, branch3x3dbl, branch_avg_pool],
                           axis=channel_axis,
                           name='mixed0')



    
    if pooling == 'avg':
        x = layers.GlobalAveragePooling2D(name='avg_pool')(x)
    elif pooling == 'max':
        x = layers.GlobalMaxPooling2D(name='max_pool')(x)
    x = layers.Dense(classes, activation=classifier_activation,
                     name='predictions')(x)
    
    if input_tensor is not None:
        inputs = layer_utils.get_source_inputs(input_tensor)
    else:
        inputs = img_input
        
    # Create model.
    model = training.Model(inputs, x, name='dashnet')
    
    return model
