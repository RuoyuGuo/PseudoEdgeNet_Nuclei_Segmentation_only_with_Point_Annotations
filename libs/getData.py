# -*- coding: utf-8 -*-
"""
Created on Mon Jun 29 23:25:40 2020

@author: Ruoyu
"""

import xml.etree.ElementTree as ET
import cv2 as cv
import numpy as np
import os
import sys

from tqdm import tqdm

def _get_centroid(cnt):
    '''
    return centroid coordinate of a nucleus
    using cv moment
    '''
    #Arguments:
    #cnt: a N * 1 * 2 numpy array, where N is the number of pixel in contour
    
    M = cv.moments(cnt)
    
    cX = 0
    cY = 0
    
    if M["m00"] != 0:
        cX = int(M["m10"] / M["m00"])
        cY = int(M["m01"] / M["m00"])
        
    return cX, cY

def _get_binary_image(path):
    '''
    read xml file from path
    return a binary image of centroids
    '''

    #Arguments:
    #path: path that contain train label(xml file)
    
    #read xml file
    tree = ET.parse(path)
    root = tree.getroot()
    annos = []
    
    #get edge annotation coordinates
    for region in root.iter('Region'):
        annos.append(region)

    cnts = []                             #store contour coordinates of each nucleus
    img = np.zeros((1000, 1000), dtype=np.bool_)
    #ctds = []                             #store centroid coordinates of each nucleus

    #get contour and centroid of each nucleus
    for region in annos:
        cnt = []

        for vertex in region.iter('Vertex'):
            cnt.append((np.float32(vertex.attrib['X']), np.float32(vertex.attrib['Y'])))

        cnt = np.array(cnt)
        cnt = cnt[:, np.newaxis, :]
        ctd = _get_centroid(cnt)

        cnts.append(cnt.astype(np.int32))
        img[ctd[0], ctd[1]] = 1
        #ctds.append(ctd)
    
    return img
    
def _get_point_annos_data(path, data_ids):
    '''
    Generate point annotation binary image
    '''
    
    #Arguments:
    #path: path that contain labels
    #data_ids: data id(without filename extension)
    
    output = np.zeros((len(data_ids), 1000, 1000), dtype=np.bool_)
    
    #write point annotation of each image 
    print('Generating point annotation...')
    sys.stdout.flush()
    for i in tqdm(range(len(data_ids)), total=len(data_ids)):
        img = _get_binary_image(os.path.join(path, data_ids[i]+'.xml'))
        output[i] = img
    print('Done!')

    return output

def _get_img_data(path, data_ids):
    '''
    read image in rgb format and return
    '''

    #Arguments:
    #path: path that contain image
    #data_ids: data id(without filename extension)
    
    output = np.zeros((len(data_ids), 1000, 1000, 3), dtype=np.uint8)
    
    #read image
    print('Reading image...')
    sys.stdout.flush()
    for i in tqdm(range(len(data_ids)), total=len(data_ids)):
        img = cv.imread(os.path.join(path, data_ids[i]+'.png'))
        img = cv.cvtColor(img, cv.COLOR_BGR2RGB)
        output[i] = img
    print('Done!')
    
    return output
    
    
def data(img_path, label_path, data_ids):
    '''
    return image data(X) and label data(Y)
    '''

    #Arguments:
    #img_path: path that contain image
    #label_path: path that contain label
    #data_ids: data id(without filename extension)

    return _get_img_data(img_path, data_ids), \
        _get_point_annos_data(label_path, data_ids)
