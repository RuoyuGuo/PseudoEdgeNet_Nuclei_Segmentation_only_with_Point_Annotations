# -*- coding: utf-8 -*-
"""
build segmentation network f
backbone: ResNet50
segmetation: FPN
"""

from tensorflow import keras
from tensorflow.keras.layers import Conv2D, BatchNormalization, Activation, UpSampling2D, Concatenate
from tensorflow.keras import Model
from libs.fpn_utils import *

ResNet50 = keras.applications.ResNet50(
    include_top=False,
    weights=None,
    input_tensor=None,
    input_shape=(1000, 1000, 3),
    pooling=None
)

def build_model(my_input_tensor=None):
    #get final layer of each convx,  where x = 2, 3, 4, 5
    uni_name = 'rs50'
    
    if my_input_tensor == None:
        input_tensor = ResNet50.input
    else:
        input_tensor = my_input_tensor
        
    stage5 = ResNet50.get_layer('conv5_block3_out').output
    stage4 = ResNet50.get_layer('conv4_block6_out').output
    stage3 = ResNet50.get_layer('conv3_block4_out').output
    stage2 = ResNet50.get_layer('conv2_block3_out').output
    
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
    
    model = Model(input_tensor, f, name='segmentation network f(ResNet50_FPN)')
    
    return model