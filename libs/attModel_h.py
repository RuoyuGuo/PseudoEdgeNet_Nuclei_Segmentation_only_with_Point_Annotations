# -*- coding: utf-8 -*-
"""
build attention model h
backbone: ResNet18
segmetation: FPN
"""


from tensorflow import keras 
from tensorflow.keras.layers import Input, Conv2D, BatchNormalization, Activation, UpSampling2D, \
                                    Add, Concatenate, ZeroPadding2D, MaxPooling2D
from tensorflow.keras import Model
from libs.fpn_utils import *

def bottleNeck(name, stage, filters, strides=1, is_conv_map=False):
    '''
    build bottleNeck in convBlock
    
    first bottleNeck use conv2d mapping (replace maxpooling), others use identity mapping
    '''
    name = name+'_block'+str(stage)
    
    def wrapper(input_tensor):
        if is_conv_map:
            shortcut = Conv2D(filters=filters, kernel_size=(3, 3), strides=strides, padding='same', name=name+'_0_conv')(input_tensor)
            shortcut = BatchNormalization(epsilon=1.001e-5, name=name+'_0_bn')(shortcut)
        else:
            shortcut = input_tensor
        
        x = Conv2D(filters=filters, kernel_size=(3, 3), strides=strides, padding='same', name=name+'_1_conv')(input_tensor)
        x = BatchNormalization(epsilon=1.001e-5, name=name+'_1_bn')(x)
        x = Activation('relu', name=name+'_1_relu')(x)            
        
        x = Conv2D(filters=filters, kernel_size=(3, 3), padding='same', name=name+'_2_conv')(x)
        x = BatchNormalization(epsilon=1.001e-5, name=name+'_2_bn')(x)
        
        x = Add(name=name+'_add')([shortcut, x])
        x = Activation('relu', name=name+'_out')(x)
        
        return x
         
    return wrapper

def convBlock(stage, blocks, filters, strides=1, is_conv_map=False):
    '''
    build conv block in ResNet
    '''
    
    #Arguments:
    #stage: index of convblock
    #blocks: number of bottleNecks in one convBlock
    #filters: number of filters
    #strides: same as strides in conv2d
    #is_conv_map: if use conv2d mapping as shortcut connection, else use identiy mapping
    
    name = 'rs18_conv' + str(stage)
    
    def wrapper(input_tensor):
        x = bottleNeck(name, 1, filters, strides, is_conv_map)(input_tensor)
        
        for i in range(2, blocks+1):
            x = bottleNeck(name, i, filters)(x)
        
        return x
        
    return wrapper

def myResNet18(input_tensor=None, input_shape=(1000, 1000, 3)):
    '''
    Build ResNet18 network
    '''
    
    if input_tensor == None:
        input_tensor = Input(shape=input_shape)
    
    #conv1
    x = input_tensor
    x = ZeroPadding2D(padding=((3, 3), (3, 3)), name='rs18_conv1_pad')(x)
    x = Conv2D(64, 7, strides=2, name='rs18_conv1_conv')(x)
    x = BatchNormalization(epsilon=1.001e-5, name='rs18_conv1_bn')(x)
    x = Activation('relu', name='rs18_conv1_relu')(x)
    
    #conv2
    x = ZeroPadding2D(padding=((1, 1), (1, 1)), name='rs18_pool1_pad')(x)
    x = MaxPooling2D(pool_size=(3, 3), strides=2, name='rs18_pool1_pool')(x)
    x = convBlock(2, 2, 64)(x)
    
    #conv3-5
    x = convBlock(3, 2, 128, 2, True)(x)
    x = convBlock(4, 2, 256, 2, True)(x)
    x = convBlock(5, 2, 512, 2, True)(x)
    
    model = Model(input_tensor, x, name='attention model h')
    
    return model

def build_model(my_input_tensor=None):
    '''
    build FPN with ResNet18 as backbone
    '''
    
    uni_name = 'rs18'
    
    if my_input_tensor == None:
        ResNet18 = myResNet18()
        input_tensor = ResNet18.input
    else:
        ResNet18 = myResNet18(my_input_tensor)
        input_tensor = my_input_tensor
        
    #get final layer of each convx,  where x = 2, 3, 4, 5
    stage5 = ResNet18.get_layer('rs18_conv5_block2_out').output
    stage4 = ResNet18.get_layer('rs18_conv4_block2_out').output
    stage3 = ResNet18.get_layer('rs18_conv3_block2_out').output
    stage2 = ResNet18.get_layer('rs18_conv2_block2_out').output
    
    #Build top down pathway
    m5 = ConvMBnReLU(uni_name, filters=256, kernel_size=(1, 1), stage='5')(stage5)
    m4 = lateral_connection(uni_name, stage='4', is_crop=True)(stage4, m5)
    m3 = lateral_connection(uni_name, stage='3', is_crop=True)(stage3, m4)
    m2 = lateral_connection(uni_name, stage='2')(stage2, m3)
    
    #Attach 3 * 3 conv layer to get final feature map
    p5 = ConvPBnReLU(uni_name, filters=128, kernel_size=(3, 3), stage='5')(m5)
    p4 = ConvPBnReLU(uni_name, filters=128, kernel_size=(3, 3), stage='4')(m4)
    p3 = ConvPBnReLU(uni_name, filters=128, kernel_size=(3, 3), stage='3')(m3)
    p2 = ConvPBnReLU(uni_name, filters=128, kernel_size=(3, 3), stage='2')(m2)
    
    #Attach 3*3, conv layer to get segmentation head
    head5 = ConvHBnReLU(uni_name, filters=128, kernel_size=(3, 3), stage='5')(p5)
    head4 = ConvHBnReLU(uni_name, filters=128, kernel_size=(3, 3), stage='4')(p4)
    head3 = ConvHBnReLU(uni_name, filters=128, kernel_size=(3, 3), stage='3')(p3)
    head2 = ConvHBnReLU(uni_name, filters=128, kernel_size=(3, 3), stage='2')(p2)
    
    #upsampling and concat
    seg5 = Upsample_crop(uni_name, size=(8, 8), cropping=((3, 3), (3, 3)), is_crop=True, name='5')(head5)
    seg4 = Upsample_crop(uni_name, size=(4, 4), cropping=((1, 1), (1, 1)), is_crop=True, name='4')(head4)
    seg3 = Upsample_crop(uni_name, size=(2, 2), name='3')(head3)
    seg2 = head2
    
    #Aggregate the segmentation feature
    if keras.backend.image_data_format() == 'channels_last':
        f_concat = Concatenate(axis=3, name=uni_name+'_'+'final_concat')([seg5, seg4, seg3, seg2])
    else:
        f_concat = Concatenate(axis=1, name=uni_name+'_'+'final_concat')([seg5, seg4, seg3, seg2])
        
    #final stage, conv layer, then Upsampling, then follow a conv layer(use sigmoid)
    f = ConvFBnReLU(uni_name, filters=128, kernel_size=(3, 3), stage='1a')(f_concat)
    f = UpSampling2D(size=(2, 2), interpolation='nearest', name=uni_name+'_'+'final_upsample_a')(f)
    f = ConvFBnReLU(uni_name, filters=64, kernel_size=(3, 3), stage='1b')(f)
    f = UpSampling2D(size=(2, 2), interpolation='nearest', name=uni_name+'_'+'final_upsample_b')(f)
    f = ConvFBnSigmoid(uni_name, filters=1, kernel_size=(3, 3), is_batch=False, stage='2')(f)
    
    model = Model(input_tensor, f, name='attention model h(ResNet18_FPN)')
    
    return model
