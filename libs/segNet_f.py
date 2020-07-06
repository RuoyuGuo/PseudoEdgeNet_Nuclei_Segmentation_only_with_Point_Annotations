# -*- coding: utf-8 -*-
"""
build segmentation network f
backbone: ResNet50
segmetation: FPN
"""

from tensorflow import keras
from tensorflow.keras.layers import Conv2D, BatchNormalization, Activation, UpSampling2D, Cropping2D, Add, Concatenate
from tensorflow.keras import Model, activations

ResNet50 = keras.applications.ResNet50(
    include_top=False,
    weights=None,
    input_tensor=None,
    input_shape=(1000, 1000, 3),
    pooling=None
)

def ConvMBnReLU(filters=256, kernel_size=(1, 1), strides=(1, 1), \
                      padding='same', momentum=0.9, stage=''):
    '''
    Build a conv2d layer + BN + relu for feature map M
    '''
    
    name = 'stage'+stage + '_m' + stage + '_l2r'
    
    def wrapper(input_tensor): 
        
        x = Conv2D(filters=filters, kernel_size=kernel_size, \
                      strides=strides, padding=padding, name=name+'_conv')(input_tensor)
        x = BatchNormalization(momentum=momentum, name=name+'_bn')(x)
        x = Activation(activation=activations.relu, name=name+'_relu')(x)
        
        return x
        
    return wrapper

def ConvPBnReLU(filters=256, kernel_size=(3, 3), strides=(1, 1), \
                      padding='same', momentum=0.9, stage=''):
    '''
    Build a conv2d layer + BN + relu for feature map P
    '''
    
    name = 'stage'+stage + '_p' + stage + '_l2r'
    
    def wrapper(input_tensor):
        
        x = Conv2D(filters=filters, kernel_size=kernel_size, \
                      strides=strides, padding=padding, name=name+'_conv')(input_tensor)
        x = BatchNormalization(momentum=momentum, name=name+'_bn')(x)
        x = Activation(activation=activations.relu, name=name+'_relu')(x)
        
        return x
        
    return wrapper

def ConvHBnReLU(filters=128, kernel_size=(3, 3), strides=(1, 1), \
                      padding='same', momentum=0.9, stage=''):
    '''
    Build a conv2d layer + BN + relu for feature map P
    '''
    
    name = 'stage'+stage + '_h' + stage + '_l2r'
    
    def wrapper(input_tensor):
        
        x = Conv2D(filters=filters, kernel_size=kernel_size, \
                      strides=strides, padding=padding, name=name+'_conv')(input_tensor)
        x = BatchNormalization(momentum=momentum, name=name+'_bn')(x)
        x = Activation(activation=activations.relu, name=name+'_relu')(x)
        
        return x
        
    return wrapper

def ConvFBnReLU(filters=128, kernel_size=(3, 3), strides=(1, 1), \
                      padding='same', momentum=0.9, stage=''):
    '''
    Build a conv2d layer + BN + relu for final segmentation
    '''
    
    name ='final_conv' + stage
    
    def wrapper(input_tensor):
        
        x = Conv2D(filters=filters, kernel_size=kernel_size, \
                      strides=strides, padding=padding, name=name+'_conv')(input_tensor)
        x = BatchNormalization(momentum=momentum, name=name+'_bn')(x)
        x = Activation(activation=activations.relu, name=name+'_relu')(x)
        
        return x
        
    return wrapper

def ConvFBnSigmoid(filters=128, kernel_size=(3, 3), strides=(1, 1), \
                      padding='same', momentum=0.9, stage='', is_batch=False):
    '''
    Build a conv2d layer + BN + sigmoid for final segmentation
    '''
    
    name ='final_conv' + stage
    
    def wrapper(input_tensor):
        
        x = Conv2D(filters=filters, kernel_size=kernel_size, \
                      strides=strides, padding=padding, name=name+'_conv')(input_tensor)
        if is_batch:
            x = BatchNormalization(momentum=momentum, name=name+'_bn')(x)
        x = Activation(activation=activations.sigmoid, name=name+'_sigmoid')(x)
        
        return x
        
    return wrapper

def Upsample_crop(size=(2, 2), cropping=((1, 0), (1, 0)), is_crop=False, name=''):
    if name.isdigit():
        name = 'stage' + name + '_seg' + name + '_l2r'

    def wrapper(input_tensor):
        x = UpSampling2D(size=size, interpolation='nearest', name=name+'_upsample')(input_tensor)
        if is_crop:
            x = Cropping2D(cropping=cropping, name=name+'_crop')(x)
        
        return x
        
    return wrapper

def lateral_connection(is_crop=False,\
                          filters=256, strides=(1, 1), \
                           padding='same', momentum=0.9, stage=''):
    
    name = 'stage'+stage + '_m' + stage + '_t2d'
    
    def wrapper(bottom_up_tensor, top_down_tensor):
        bottom_up_tensor = ConvMBnReLU(filters=filters, strides=strides, padding=padding, \
                                             momentum=momentum, stage=stage)(bottom_up_tensor)
        
        top_down_tensor = Upsample_crop(size=(2, 2), cropping=((1, 0), (1, 0)), is_crop=is_crop, name=name)(top_down_tensor)

        x = Add(name=name+'_add')([bottom_up_tensor, top_down_tensor])
        
        return x
    
    return wrapper

def build_model():
    #get final layer of each convx,  where x = 2, 3, 4, 5
    input_tensor = ResNet50.input
    stage5 = ResNet50.get_layer('conv5_block3_out').output
    stage4 = ResNet50.get_layer('conv4_block6_out').output
    stage3 = ResNet50.get_layer('conv3_block4_out').output
    stage2 = ResNet50.get_layer('conv2_block3_out').output
    
    #Build top down pathway
    m5 = ConvMBnReLU(filters=256, kernel_size=(1, 1), stage='5')(stage5)
    m4 = lateral_connection(stage='4', is_crop=True)(stage4, m5)
    m3 = lateral_connection(stage='3', is_crop=True)(stage3, m4)
    m2 = lateral_connection(stage='2')(stage2, m3)
    
    #Attach 3 * 3 conv layer to get final feature map
    p5 = ConvPBnReLU(filters=256, kernel_size=(3, 3), stage='5')(m5)
    p4 = ConvPBnReLU(filters=256, kernel_size=(3, 3), stage='4')(m4)
    p3 = ConvPBnReLU(filters=256, kernel_size=(3, 3), stage='3')(m3)
    p2 = ConvPBnReLU(filters=256, kernel_size=(3, 3), stage='2')(m2)
    
    #Attach 3*3, conv layer to get segmentation head
    head5 = ConvHBnReLU(filters=256, kernel_size=(3, 3), stage='5')(p5)
    head4 = ConvHBnReLU(filters=256, kernel_size=(3, 3), stage='4')(p4)
    head3 = ConvHBnReLU(filters=256, kernel_size=(3, 3), stage='3')(p3)
    head2 = ConvHBnReLU(filters=256, kernel_size=(3, 3), stage='2')(p2)
    
    #upsampling and concat
    seg5 = Upsample_crop(size=(8, 8), cropping=((3, 3), (3, 3)), is_crop=True, name='5')(head5)
    seg4 = Upsample_crop(size=(4, 4), cropping=((1, 1), (1, 1)), is_crop=True, name='4')(head4)
    seg3 = Upsample_crop(size=(2, 2), name='3')(head3)
    seg2 = head2
    
    #Aggregate the segmentation feature
    if keras.backend.image_data_format() == 'channels_last':
        f_concat = Concatenate(axis=3, name='final_concat')([seg5, seg4, seg3, seg2])
    else:
        f_concat = Concatenate(axis=1, name='final_concat')([seg5, seg4, seg3, seg2])
        
    #final stage, conv layer, then Upsampling, then follow a conv layer(use sigmoid)
    f = ConvFBnReLU(filters=128, kernel_size=(3, 3), stage='1')(f_concat)
    f = UpSampling2D(size=(4, 4), interpolation='nearest', name='final_upsample')(f)
    f = ConvFBnSigmoid(filters=1, kernel_size=(3, 3), is_batch=False, stage='2')(f)
    
    model = Model(input_tensor, f, name='segmentation network f(ResNet50_FPN)')
    
    return model