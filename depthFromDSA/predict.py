import tensorflow as tf
import os
import sys
import numpy as np
import random
import matplotlib.pyplot as plt

import utils as ut

# generate random integer values
from random import seed
from random import randint

from tqdm import tqdm
from itertools import chain
 
from skimage.io import imread, imshow
from skimage.transform import resize

from tensorflow.python.keras.models import Sequential, Model, load_model
from tensorflow.python.keras.layers import Dense, Dropout, Activation, Flatten, Input
from tensorflow.python.keras.layers import Conv2D, MaxPooling2D, Lambda, Conv2DTranspose, concatenate
from tensorflow.python.keras.callbacks import EarlyStopping, ModelCheckpoint
from keras import backend as K

seed(1)

IMG_SIZE = 128
IMG_CHANNELS = 1
N_CLASSES = 50 # 50 shades of grey
SET_LENGHT = -1 # -1 for all

# the directory train/ contains the directory images/ and masks/
TRAIN_RAW_PATH = 'xray/'
TRAIN_MASK_PATH = 'depth/'


x_train = ut.loadRawData(TRAIN_RAW_PATH, IMG_SIZE, SET_LENGHT)
x_train = x_train/255.
x_train = np.expand_dims(x_train, axis=3)

y_train = ut.loadMaskData(TRAIN_MASK_PATH, IMG_SIZE, N_CLASSES, SET_LENGHT)


print('Data loaded')
print(x_train.shape)
print(y_train.shape)

    
#Predict on test set
model = load_model('model-mini-elu-nf32-ep25-bs32-bnf-128.h5', custom_objects={'iou_coef': ut.iou_coef, 'dice_coef': ut.dice_coef})



fig = plt.figure(figsize=(40,8))    

figCols = 3
figRows = 6
for i in range(0,figRows):
    
    ix = randint(0, x_train.shape[0]-1)    
    p_train = model.predict(np.expand_dims(x_train[ix], axis=0), verbose=1)
    
    ax1 = fig.add_subplot(figRows,figCols,i*figCols+1)
    ax1.imshow(np.squeeze(x_train[ix]))
    ax1.set_title('Actual frame')  #input
    
    ax2 = fig.add_subplot(figRows,figCols,i*figCols+2)
    ax2.set_title('Ground truth labels')
    ax2.imshow(np.argmax(y_train[ix], axis=-1))  #ground truth

    ax3 = fig.add_subplot(figRows,figCols,i*figCols+3)
    ax3.set_title('Predicted labels')
    ax3.imshow(np.squeeze(np.argmax(p_train, axis=-1))) #prediction
    
plt.show()
