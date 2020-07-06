# -*- coding: utf-8 -*-
"""
build pseuduEdgeNet g
"""

from tensorflow import keras
from tensorflow.keras.layers import Conv2D, BatchNormalization, Activation, \
                                        UpSampling2D, Cropping2D, Add, Concatenate
from tensorflow.keras import Model, activations

def build_model(input_tensor,\
                num_of_layers=4, \
                input_shapes=(1000, 1000, 3), \
                filters=[64, 64, 64, 2], \
                kernel_sizes=[(3, 3), (3, 3), (3, 3), (3, 3)], \
                is_batchs=[True, True, True, False], \
                momentum=0.9, \
                is_activation=[True, True, True, False] \
               ):

    '''
    retrun PseudoEdgeNet model
    '''
    
    if input_tensor == None:
        input_tensor = keras.Input(shape=input_shapes)

    x = input_tensor
    for i in range(num_of_layers):
        x = Conv2D(filters=filters[i], kernel_size=kernel_sizes[i], \
                   strides=(1, 1), padding='same', name=f'pseNet_conv{i+1}_conv')(x)
        if is_batchs[i]:
            x = BatchNormalization(momentum=momentum, name=f'pseNet_conv{i+1}_bn')(x)
        if is_activation[i]:
            x = Activation(activation=activations.relu, name=f'pseNet_conv{i+1}_relu')(x)
    
    model = Model(input_tensor, x, name='PesudoEdgeNet g')
    
    return model