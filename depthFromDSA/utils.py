import os
import sys
import numpy as np
import skimage as sk

from tqdm import tqdm
from itertools import chain
 
from skimage.io import imread, imshow
from skimage.transform import resize

from tensorflow.python.keras.models import Sequential, Model
from tensorflow.python.keras.layers import Dense, Dropout, Activation, Flatten, Input
from tensorflow.python.keras.layers import Conv2D, MaxPooling2D, Lambda, Conv2DTranspose, concatenate
from tensorflow.python.keras.callbacks import EarlyStopping, ModelCheckpoint
from keras import backend as K




def iou_coef(y_true, y_pred, smooth=1):
    intersection = K.sum(K.abs(y_true * y_pred), axis=[1,2,3])
    union = K.sum(y_true,[1,2,3])+K.sum(y_pred,[1,2,3])-intersection
    iou = K.mean((intersection + smooth) / (union + smooth), axis=0)
    return iou

def dice_coef(y_true, y_pred, smooth=1):
    intersection = K.sum(y_true * y_pred, axis=[1,2,3])
    union = K.sum(y_true, axis=[1,2,3]) + K.sum(y_pred, axis=[1,2,3])
    dice = K.mean((2. * intersection + smooth)/(union + smooth), axis=0)
    return dice


def loadRawData(disk_path, size, lenght = -1):    
    sys.stdout.flush()
    nbImg = len(os.listdir(disk_path))
    if (lenght == -1 or lenght > nbImg):
        lenght = nbImg
    
    x = np.zeros((lenght, size, size), dtype=np.uint8)

    for n, img in enumerate(tqdm(os.listdir(disk_path))):
        if n == lenght:
            break
        path = os.path.join(disk_path,img)
        img = imread(path)
        img[img==255]=0 # replace white pixels with black pixels
        img = resize(img, (size, size), mode='constant', preserve_range=True).astype(np.uint8)
        x[n] = img
        
    return x


def tensorFromMask(mask, depth):
    height = mask.shape[0]
    width = mask.shape[1]
    
    tens = np.zeros((height, width, depth), dtype=np.int8) # height X width X depth
    for i in range(0, height): # We go over rows number, height 
        for j in range(0, width): # we go over columns number, width
            nc = mask[i,j] # the value of the class is the pixel intensity
            tens[i,j,nc] = 1 # create the mask
            
    return tens


def loadMaskData(disk_path, size, depth, lenght = -1):    
    sys.stdout.flush()
    nbImg = len(os.listdir(disk_path))
    if (lenght == -1 or lenght > nbImg):
        lenght = nbImg
    
    x = np.zeros((lenght, size, size, depth), dtype=np.int8)

    for n, img in enumerate(tqdm(os.listdir(disk_path))):        
        if n == lenght:
            break
        path = os.path.join(disk_path,img)
        img = imread(path)
        img[img==255]=0 # replace white pixels with black pixels
        img = resize(img, (size, size), mode='constant', preserve_range=True).astype(np.int8)
        tens = tensorFromMask(img, depth)
        x[n] = tens
            
    return x


def maskFromTensor(tensor):
    # this will find the max depth over all depth values
    mask = np.argmax(tensor, axis=-1) # height X width X 1
    return mask
         