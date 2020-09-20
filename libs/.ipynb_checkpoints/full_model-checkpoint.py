# -*- coding: utf-8 -*-
"""
build full model with attention map
"""

from libs import pseNet_g as g
from libs import segNet_f as f
from libs import attModel_h as h

from tensorflow import keras
from tensorflow.keras import Model
from tensorflow.keras.layers import Subtract, Reshape, Multiply
from tensorflow.keras import backend as K

import tensorflow as tf

# custom loss function
#def my_loss1(): 
def loss1(y_true, y_pred):
#first term
#only consider pixels that are annotated
    y_true = K.clip(y_true, K.epsilon(), 1-K.epsilon())
    y_pred = K.clip(y_pred, K.epsilon(), 1-K.epsilon())

    #get point and boundary annotation separately
    y_true_pos = y_true[:,:,:,0]
    y_true_neg = y_true[:,:,:,1]

    #flatten, (batch, pixels)
    y_true_pos_f = K.batch_flatten(y_true_pos)
    y_true_neg_f = K.batch_flatten(y_true_neg)
    y_pred_f = K.batch_flatten(y_pred)

    #only consider annotated pixels
    #(batch, )
    y_true_pos_count = K.sum(y_true_pos_f, axis=-1)
    y_true_neg_count = K.sum(y_true_neg_f, axis=-1)

    #cross_entropy of each image
    #(batch, )
    cross_entropy_pos = K.sum(-y_true_pos_f * K.log(y_pred_f), axis=-1)
    cross_entropy_neg = K.sum(-y_true_neg_f * K.log(1-y_pred_f), axis=-1)

    #loss_pos = K.mean(cross_entropy_pos / y_true_pos_count)
    #loss_neg = K.mean(cross_entropy_neg / y_true_neg_count)

    loss_pos = cross_entropy_pos / y_true_pos_count
    loss_neg = cross_entropy_neg / y_true_neg_count

    return 1.0 * loss_pos + 0.1 * loss_neg 
        
    #return loss 

#def my_loss2(lambda_value):
#second term 
def loss2(y_true, y_pred):
    return K.mean(K.batch_flatten(K.abs(y_pred)), axis=-1)

#return loss

def my_IoU(y_true, y_pred):
#IOU metrics
#1 for object, 0 for background
    y_true = y_true[:,:,:,2]
    smooth=1
    
    #threshold to determine positive pixel
    y_pred = K.cast_to_floatx(K.greater_equal(y_pred, 0.5))
    
    #flatten, (batch, pixels)
    y_true_f = K.batch_flatten(y_true)
    y_pred_f = K.batch_flatten(y_pred)
    
    #(batchs)
    intersection = K.sum(y_true_f * y_pred_f, axis=-1)
    union = K.sum(y_pred_f, axis=-1) + K.sum(y_true_f, axis=-1) - intersection
    
    return (intersection + smooth) / (union + smooth)
    

def get_model():
    segNet = f.build_model()

    input_tensor = segNet.input
    pseNet = g.build_model(input_tensor)
    attModel = h.build_model(input_tensor)
    
    #combine three network
    sobel_tensor = tf.image.sobel_edges(segNet.output)
    reshape_tensor = Reshape((1000, 1000, 2), name='second_term_reshape')(sobel_tensor)
    multiply_tensor = Multiply(name='second_term_multiply')([pseNet.output, attModel.output])
    sub_tensor = Subtract(name='second_term_sub')([reshape_tensor, multiply_tensor])
    model = Model(input_tensor, [segNet.output, sub_tensor], name='pseudoEdge_FPN')
    
    return model