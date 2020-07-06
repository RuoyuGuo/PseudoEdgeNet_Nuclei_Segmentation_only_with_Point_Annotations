# -*- coding: utf-8 -*-
"""
build full model without attention map
"""

from libs import pseNet_g as g
from libs import segNet_f as f

from tensorflow import keras
from tensorflow.keras import Model
from tensorflow.keras.layers import Subtract, Reshape
from tensorflow.keras import backend as K

import tensorflow as tf

# custom loss function
def my_loss1(): 
    def loss(y_true, y_pred):
    #first term
    #only consider pixels that are annotated
        
        y_true = K.clip(y_true, K.epsilon(), 1-K.epsilon())
        y_pred = K.clip(y_pred, K.epsilon(), 1-K.epsilon())
        width = y_pred.shape[1]
        height = y_pred.shape[2]
        re = width*height
        
        #get point and boundary annotation separately
        y_true_pos = y_true[:,:,:,0]
        y_true_neg = y_true[:,:,:,1]

        y_true_pos_f = K.reshape(y_true_pos, shape=(-1, re))
        y_true_neg_f = K.reshape(y_true_neg, shape=(-1, re))
        y_pred_f = K.reshape(y_pred, shape=(-1, re))

        #only consider annotated pixels
        y_true_pos_count = K.sum(y_true_pos_f)
        y_true_neg_count = K.sum(y_true_neg_f)

        loss_pos = K.sum(-y_true_pos_f * K.log(y_pred_f)) / y_true_pos_count
        loss_neg = K.sum(-y_true_pos_f * K.log(y_pred_f)) / y_true_neg_count

        return 1.0 * loss_pos + 0.1 * loss_neg 
        
    return loss

def my_loss2(lambda_value):
#second term 
    def loss(y_true, y_pred):
        return lambda_value * K.mean(K.abs(y_pred))
    
    return loss

def get_model():
    segNet = f.build_model()

    input_tensor = segNet.input
    pseNet = g.build_model(input_tensor)
    
    #combine two network
    sobel_tensor = tf.image.sobel_edges(segNet.output)
    reshape_tensor = Reshape((1000, 1000, 2), name='second_term_reshape')(sobel_tensor)
    sub_tensor = Subtract(name='second_term_sub')([reshape_tensor, pseNet.output])
    model = Model(input_tensor, [segNet.output, sub_tensor], name='pseudoEdge_FPN')
    
    return model