'''
Data Augmentation
'''

import cv2 as cv
import numpy as np
import imgaug as ia
import imgaug.augmenters as iaa


print(ia.__version__)

def myDataAug(X_data, Y_data, its, seed=12):
    size = len(X_data)
    ia.seed(seed)

    seq = iaa.Sequential([
        iaa.GaussianBlur(sigma=(0.0, 4.0)),    #Random GaussianBlur from sigma 0 to 4
        #iaa.AddToHueAndSaturation((-10, 10), per_channel=True),      #color jittering
        iaa.Affine(translate_percent={"x": (-0.1, 0.1), "y": (-0.1, 0.1)}),   #affine transalte (affects segmaps)
        iaa.Fliplr(0.5),              #50 % to flip horizontally (affects segmaps)
        iaa.Flipud(0.5),              #50 % to flip vertically  (affects segmaps)
        iaa.Rotate((-45, 45))  # rotate by -45 to 45 degrees (affects segmaps)
    ], random_order=True)

    X_data_augs = np.zeros((its*size, 1000, 1000, 3), dtype=np.uint8)
    Y_data_augs = np.zeros((its*size, 1000, 1000, 3), dtype=np.uint8)

    for i in range(its):
        X_data_aug, Y_data_aug = seq(images=X_data, segmentation_maps=Y_data)
        X_data_augs[i*size: (i+1)*size] = X_data_aug
        Y_data_augs[i*size: (i+1)*size] = Y_data_aug

    return X_data_augs, Y_data_augs
