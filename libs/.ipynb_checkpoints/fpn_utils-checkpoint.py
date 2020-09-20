# -*- coding: utf-8 -*-
"""
function for building FPN
"""

from tensorflow.keras.layers import Conv2D, BatchNormalization, Activation, UpSampling2D, Cropping2D, Add
from tensorflow.keras import activations

def ConvMBnReLU(uni_name, filters=256, kernel_size=(1, 1), strides=(1, 1), \
                      padding='same', stage='', use_bn=False, use_act=False):
    '''
    Build a conv2d layer + BN + relu for feature map M
    '''
    
    name = uni_name+'_'+'stage'+stage + '_m' + stage + '_l2r'
    
    def wrapper(input_tensor): 
        
        x = Conv2D(filters=filters, kernel_size=kernel_size, \
                      strides=strides, padding=padding, name=name+'_conv')(input_tensor)
        if use_bn == True:
            x = BatchNormalization(name=name+'_bn')(x)
        if use_act == True:
            x = Activation(activation=activations.relu, name=name+'_relu')(x)
        
        return x
        
    return wrapper

def ConvPBnReLU(uni_name, filters=256, kernel_size=(3, 3), strides=(1, 1), \
                      padding='same', stage=''):
    '''
    Build a conv2d layer + BN + relu for feature map P
    '''
    
    name = uni_name+'_'+'stage'+stage + '_p' + stage + '_l2r'
    
    def wrapper(input_tensor):
        
        x = Conv2D(filters=filters, kernel_size=kernel_size, \
                      strides=strides, padding=padding, name=name+'_conv')(input_tensor)
        x = BatchNormalization(name=name+'_bn')(x)
        x = Activation(activation=activations.relu, name=name+'_relu')(x)
        
        return x
        
    return wrapper

def ConvHBnReLU(uni_name, filters=128, kernel_size=(3, 3), strides=(1, 1), \
                      padding='same', stage=''):
    '''
    Build a conv2d layer + BN + relu for feature map P
    '''
    
    name = uni_name+'_'+'stage'+stage + '_h' + stage + '_l2r'
    
    def wrapper(input_tensor):
        
        x = Conv2D(filters=filters, kernel_size=kernel_size, \
                      strides=strides, padding=padding, name=name+'_conv')(input_tensor)
        x = BatchNormalization(name=name+'_bn')(x)
        x = Activation(activation=activations.relu, name=name+'_relu')(x)
        
        return x
        
    return wrapper

def ConvFBnReLU(uni_name, filters=128, kernel_size=(3, 3), strides=(1, 1), \
                      padding='same', stage=''):
    '''
    Build a conv2d layer + BN + relu for final segmentation
    '''
    
    name = uni_name+'_'+'final_conv' + stage
    
    def wrapper(input_tensor):
        
        x = Conv2D(filters=filters, kernel_size=kernel_size, \
                      strides=strides, padding=padding, name=name+'_conv')(input_tensor)
        x = BatchNormalization(name=name+'_bn')(x)
        x = Activation(activation=activations.relu, name=name+'_relu')(x)
        
        return x
        
    return wrapper

def ConvFBnSigmoid(uni_name, filters=128, kernel_size=(3, 3), strides=(1, 1), \
                      padding='same', stage='', is_batch=False):
    '''
    Build a conv2d layer + BN + sigmoid for final segmentation
    '''
    
    name = uni_name+'_'+'final_conv' + stage
    
    def wrapper(input_tensor):
        
        x = Conv2D(filters=filters, kernel_size=kernel_size, \
                      strides=strides, padding=padding, name=name+'_conv')(input_tensor)
        if is_batch:
            x = BatchNormalization(name=name+'_bn')(x)
        x = Activation(activation=activations.sigmoid, name=name+'_sigmoid')(x)
        
        return x
        
    return wrapper

def Upsample_crop(uni_name, size=(2, 2), cropping=((1, 0), (1, 0)), is_crop=False, name=''):
    if name.isdigit():
        name = uni_name + '_' +'stage' + name + '_seg' + name + '_l2r'
    else:
        name = uni_name + '_' + name
    
    def wrapper(input_tensor):
        x = UpSampling2D(size=size, interpolation='nearest', name=name+'_upsample')(input_tensor)
        if is_crop:
            x = Cropping2D(cropping=cropping, name=name+'_crop')(x)
        
        return x
        
    return wrapper

def lateral_connection(uni_name, is_crop=False,\
                          filters=256, strides=(1, 1), \
                           padding='same', stage=''):
    
    name = uni_name +'_'+ 'stage'+stage + '_m' + stage + '_t2d'
    
    def wrapper(bottom_up_tensor, top_down_tensor):
        bottom_up_tensor = ConvMBnReLU(uni_name, filters=filters, strides=strides, padding=padding, \
                                             stage=stage)(bottom_up_tensor)
        
        top_down_tensor = Upsample_crop(uni_name, size=(2, 2), cropping=((1, 0), (1, 0)), is_crop=is_crop, name=name)(top_down_tensor)

        x = Add(name=name+'_add')([bottom_up_tensor, top_down_tensor])
        
        return x
    
    return wrapper

