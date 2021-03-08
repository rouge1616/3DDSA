import tensorflow as tf
import os
import sys
import numpy as np
import random
import math
import matplotlib.pyplot as plt

from PIL import Image, ImageOps

import utils as ut
import unet as un

from tqdm import tqdm
from itertools import chain

from random import seed
from random import randint

 
from skimage.io import imread, imshow
from skimage.transform import resize

from tensorflow.python.keras.models import Sequential, Model
from tensorflow.python.keras.layers import Dense, Dropout, Activation, Flatten, Input
from tensorflow.python.keras.layers import Conv2D, MaxPooling2D, Lambda, Conv2DTranspose, concatenate
from tensorflow.python.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from tensorflow.python.keras.preprocessing.image import ImageDataGenerator
from keras import backend as K
#from keras.optimizers import Adam
from sklearn.model_selection import train_test_split
 
#tf.enable_eager_execution()
 
seed(1)    
    
IMG_SIZE = 128
IMG_CHANNELS = 1
N_CLASSES = 50 # 50 shades of grey
SET_LENGHT = -1 # -1 for all

# the directory train/ contains the directory images/ and masks/
TRAIN_RAW_PATH = 'xray/'
TRAIN_MASK_PATH = 'depth/'


x = ut.loadRawData(TRAIN_RAW_PATH, IMG_SIZE, SET_LENGHT)
y = ut.loadMaskData(TRAIN_MASK_PATH, IMG_SIZE, N_CLASSES, SET_LENGHT)

x = x/255.
x = np.expand_dims(x, axis=3)

print('Data loaded')
print(x.shape)
print(y.shape)

# Split train and valid
x_train, x_valid, y_train, y_valid = train_test_split(x, y, test_size=0.1)

print('Data splited')
print(x_train.shape)
print(y_train.shape)

# Check if training data looks all right
fig = plt.figure(figsize=(40,8))    

figCols = 2
figRows = 8
for i in range(0,figRows):
    
    ix = randint(0, x_train.shape[0]-1)    
    
    ax1 = fig.add_subplot(figRows,figCols,i*figCols+1)
    ax1.imshow(np.squeeze(x_train[ix]))
    ax1.set_title('Actual frame')  #input
    
    ax2 = fig.add_subplot(figRows,figCols,i*figCols+2)
    ax2.set_title('Ground truth labels')
    ax2.imshow(np.squeeze(y_train[ix,:,:,10])) # mask of class 10        
    
    
plt.show()


model = un.miniUnet(IMG_SIZE, IMG_SIZE, IMG_CHANNELS, N_CLASSES)
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=["accuracy", ut.iou_coef, ut.dice_coef])
model.summary()


callbacks = [
    EarlyStopping(patience=10, verbose=1),
    ReduceLROnPlateau(factor=0.1, patience=5, min_lr=0.00001, verbose=1),
    ModelCheckpoint('model-mini-elu-nf32-ep25-bs32-bnf-128.h5', verbose=1, save_best_only=True)
]

results = model.fit(x_train, y_train, batch_size=32, epochs=25, callbacks=callbacks, validation_data=(x_valid, y_valid))


# Evaluate on validation set (this must be equals to the best log_loss)
print(model.evaluate(x_valid, y_valid, verbose=1))


# Get actual number of epochs model was trained for
N = len(results.history['loss'])

#Plot the model evaluation history
plt.style.use("ggplot")
fig = plt.figure(figsize=(40,8))

fig.add_subplot(1,4,1)
plt.title("Training Loss")
plt.plot(np.arange(0, N), results.history["loss"], label="train_loss")
plt.plot(np.arange(0, N), results.history["val_loss"], label="val_loss")
plt.plot(np.argmin(results.history["val_loss"]), np.min(results.history["val_loss"]), marker="x", color="r",label="best model")

fig.add_subplot(1,4,2)
plt.title("Training Accuracy")
plt.plot(np.arange(0, N), results.history["acc"], label="train_accuracy")
plt.plot(np.arange(0, N), results.history["val_acc"], label="val_accuracy")

plt.xlabel("Epoch #")
plt.ylabel("Loss/Accuracy")
plt.legend(loc="lower left")

fig.add_subplot(1,4,3)
plt.title("Training Accuracy")
plt.plot(np.arange(0, N), results.history["iou_coef"], label="train_iou")
plt.plot(np.arange(0, N), results.history["val_iou_coef"], label="val_iou")

plt.xlabel("Epoch #")
plt.ylabel("Loss/Accuracy")
plt.legend(loc="lower left")


fig.add_subplot(1,4,4)
plt.title("Training Accuracy")
plt.plot(np.arange(0, N), results.history["dice_coef"], label="train_dice")
plt.plot(np.arange(0, N), results.history["val_dice_coef"], label="val_dice")

plt.xlabel("Epoch #")
plt.ylabel("Loss/Accuracy")
plt.legend(loc="lower left")


plt.show()

