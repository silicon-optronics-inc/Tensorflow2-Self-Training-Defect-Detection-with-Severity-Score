# -*- coding: utf-8 -*-
"""
Created on Thu Jul  2 16:50:42 2020
# https://www.programmersought.com/article/60001511310/
# https://github.com/Tony607/Focal_Loss_Keras/blob/master/src/keras_focal_loss.ipynb
@author: martinchen
"""
import tensorflow as tf
from tensorflow.keras import backend as K



def category_focal_loss1(alpha, gamma=2.0):
    """focal loss for multi categories
        
        Arguments:
            alpha: Specify the weights of each catgory, the length of array should be same as the number of categories.
            gamma: Set gamma greater than 0 to focus on training hard, misclassified examples
        
        Usage:
            loss_func = [category_focal_loss1(alpha=[1,2], gamma=2)]
            model.compile(optimizer=OPTIMIZER, loss=loss_func, metrics=['accuracy'])
    """
    
    epsilon = 1.e-7
    alpha = tf.constant(alpha, dtype=tf.float32)
    gamma = float(gamma)
    
    def category_focal_loss_fixed(y_true, y_pred):
        y_true = tf.cast(y_true, tf.float32)
        y_pred = tf.clip_by_value(y_pred, epsilon, 1. - epsilon)
        ce = -tf.multiply(y_true, tf.math.log(y_pred))
        weight = tf.multiply(y_true, tf.pow(tf.subtract(1., y_pred), gamma))
        fl = tf.multiply(alpha, tf.multiply(weight, ce))
        loss = tf.reduce_mean(fl)
        return loss
    return category_focal_loss_fixed
    

