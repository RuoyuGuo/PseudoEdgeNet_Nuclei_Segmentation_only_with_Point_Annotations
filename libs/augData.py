# -*- coding: utf-8 -*-
"""
data augmentation
"""

import numpy as np
import cv2 as cv


def Color_jittering(X_data, Y_data, seed=12):
#random shift some pixels around
#only change x data, fix y data

    np.random.seed(seed)
    noise = np.random.randint(-20, 21, X_data.shape)
    new_X_data = np.clip(X_data + noise, 0, 255).astype(np.uint8)
    
    return new_X_data, Y_data

def Gaussian_blur(X_data, Y_data, sigma=1.2):
#apply Gaussian filter on each image    
#only change x data, fix y data

    new_X_data = np.zeros(X_data.shape, dtype=np.uint8)
    
    for i in range(len(X_data)):
        new_img = cv.GaussianBlur(X_data[i], (0,0), sigma)
        new_X_data[i] = new_img
    
    return new_X_data, Y_data

def Gaussian_noise(X_data, Y_data, shape=(1000, 1000, 3), sigma=20):
#add Gausian noise on each image    
#only change x data, fix y data
    noise = np.zeros(shape, dtype=np.uint8)
    noise = cv.randn(noise, 0, sigma)
    new_X_data = X_data + noise
    
    return new_X_data, Y_data

def Rotation(X_data, Y_data, shape=(1000, 1000), angle=40, seed=12):
#rotate image, using random generate angle
#both change x data and y data.
    np.random.seed(seed)
    new_X_data = np.zeros(X_data.shape, dtype=np.uint8)
    new_Y_data = np.zeros(Y_data.shape, dtype=np.uint8)
    
    center = shape[0]//2, shape[1]//2
    
    #randomly rotate each image
    for i in range(len(X_data)):
        rand_angle = np.random.randint(-angle, angle)
        M = cv.getRotationMatrix2D(center, rand_angle, 1)
        new_X_img = cv.warpAffine(X_data[i], M, shape)
        new_Y_img = cv.warpAffine(Y_data[i], M, shape)
        
        new_X_data[i] = new_X_img
        new_Y_data[i] = new_Y_img
    
    return new_X_data, new_Y_data

def Vertical_flip(X_data, Y_data):
#vertical flip image
#0 vertical flip
    
    new_X_data = np.zeros(X_data.shape, dtype=np.uint8)
    new_Y_data = np.zeros(Y_data.shape, dtype=np.uint8)
    
    #flip each image
    for i in range(len(X_data)):
        new_X_img = cv.flip(X_data[i], 0)
        new_Y_img = cv.flip(Y_data[i], 0)
        
        new_X_data[i] = new_X_img
        new_Y_data[i] = new_Y_img
    
    return new_X_data, new_Y_data

def Horizontal_flip(X_data, Y_data):
#horizontal flip image
#1 horizontal flip
    new_X_data = np.zeros(X_data.shape, dtype=np.uint8)
    new_Y_data = np.zeros(Y_data.shape, dtype=np.uint8)
    
    #flip each image
    for i in range(len(X_data)):
        new_X_img = cv.flip(X_data[i], 1)
        new_Y_img = cv.flip(Y_data[i], 1)
        
        new_X_data[i] = new_X_img
        new_Y_data[i] = new_Y_img
    
    return new_X_data, new_Y_data


def Aug_data(X_data, Y_data, 
             num = 30,
             color_jittering=12, 
             gaussian_blur=1.2,
             gaussian_noise=20,
             rotation=40,
             vertical_flip=True,
             horizontal_flip=True,
             seed=12,
             count=7):
    '''
    return augmented new data set including original data
    '''
    
    #Arguments:
    #num: number of data
    #shape: img size
    #color_jittering: int, jittering seed
    #gaussian_blur: float, sigma, standard deviation of gaussian distribution
    #gaussian_noise: float, sigma, standard deviation of gaussian distribution
    #rotation: int, angle, rotation range from [-angle, angle]
    #flip: bool, 0 or 1
    #count: number of used augmentation techs
    
    '''
    new_X_data[30:60], new_Y_data[30:60] = \
                Color_jittering(X_data, Y_data, seed=seed)
    
    new_X_data[60:90], new_Y_data[60:90] = \
                Gaussain_blur(X_data, Y_data, shape=shape, sigma=gaussian_blur)
    
    new_X_data[90:120], new_Y_data[90:120] = \
                Gaussain_noise(X_data, Y_data, shape=shape, sigma=gaussian_noise)
    
    new_X_data[120:150], new_Y_data[120:150] = \
                Rotation(X_data, Y_data, shape=shape, angle=rotation, seed=seed)
    
    new_X_data[150:180], new_Y_data[150:180] = \
                Vertical_flip(X_data, Y_data)
        
    new_X_data[180:], new_Y_data[180:] = \
                Horizontal_flip(X_data, Y_data)
    '''
    
    datagen = [Color_jittering, Gaussian_blur, Gaussian_noise, 
              Rotation, Vertical_flip, Horizontal_flip]
    
    X_shape = list(X_data.shape)
    Y_shape = list(Y_data.shape)
    X_shape[0] = X_shape[0] * count
    Y_shape[0] = Y_shape[0] * count
    
    new_X_data = np.zeros(X_shape, dtype=np.uint8)
    new_Y_data = np.zeros(Y_shape, dtype=np.uint8)
    
    new_X_data[:num] = X_data
    new_Y_data[:num] = Y_data
    
    for i in range(len(datagen)):
        new_X_data[num*(i+1): num*(i+2)], new_Y_data[num*(i+1): num*(i+2)] = \
                datagen[i](X_data, Y_data)
    
    return new_X_data, new_Y_data