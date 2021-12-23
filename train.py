#!/usr/bin/env python
# coding: utf-8



import numpy as np 
import os
import sys
import skimage.io as io
import skimage.transform as trans
import numpy as np

import keras
from keras.preprocessing.image import ImageDataGenerator
import matplotlib.image as mpimg
from PIL import Image 
import numpy as np
import CNN_model
import Unet_model
import tensorflow_addons as tfa
import data
#from unet_model import unet
from helpers import *

def train_model(model_type):
    NUM_EPOCHS=20
    BATCH_SIZE = 16  # 64
    NUM_CHANNELS = 3  # RGB images
    PIXEL_DEPTH = 255
    NUM_LABELS = 2
    IMG_PATCH_SIZE = 16    
    data_dir='drive/MyDrive/LouayML/training'
    ##Create Train and validation Sets
    data_gen_args = dict(
            rotation_range=180,
            horizontal_flip=True,
            vertical_flip=True,)

    X_train, X_val, y_train, y_val = data.preprocessing(BATCH_SIZE, data_dir, data_gen_args,split_ratio = 0.2)  
    
    
##Train the models

    if (model_type=='CNN'):
        model=CNN_model.CNN()
        
        data_dir = 'data/training/'
        
        initial_learning_rate = 1e-3
        lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(
                        initial_learning_rate,
                        decay_steps=100000,
                        decay_rate=0.6,
                        staircase=True)
        model.compile( optimizer=keras.optimizers.Adam(1e-4),
                    loss="binary_crossentropy",
                    metrics=[tfa.metrics.F1Score(num_classes=2)])
        X_train_,y_train_ = extract_data_pathches(X_train,y_train)
        X_val_,y_val_ = extract_data_pathches(X_val,y_val)
        history = model.fit(X_train_, y_train_, epochs=NUM_EPOCHS, validation_data= (X_val_, y_val_))
        return model
    
    
    if (model_type=='Unet'):
        model = Unet_model.unet()
        # Callback to save model weights
        model.compile( optimizer=keras.optimizers.Adam(1e-4),
                loss="binary_crossentropy",
                metrics=[tfa.metrics.F1Score(num_classes=2)])
    
        train_path = '/content/drive/MyDrive/ML_Road_Segmentation/training'        
        history = model.fit(X_train,y_train,epochs=NUM_EPOCHS,validation_data=(X_val,y_val),callbacks=model_callbacks)
        
        return model 

