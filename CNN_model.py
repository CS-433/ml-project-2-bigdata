#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import numpy as np 
import os
import sys
import skimage.io as io
import skimage.transform as trans
import numpy as np

import tensorflow as tf 
from tensorflow import keras
from tensorflow.keras import layers,regularizers
from keras.regularizers import l2
from keras.models import Model

from tensorflow.keras.utils import to_categorical
from tensorflow.keras import regularizers
import torch

def CNN():
    leak_alpha=0.01
    dropout_prob = 0.2 
    regularization_value = 1e-6
    model = keras.Sequential()

    # Input layer
    input_shape = ( 16,  16,  3)
    model.add(layers.InputLayer(input_shape))

    # First convolution layer : 5x5 filter, depth 64
    model.add(layers.Conv2D(32, 5, padding='same'))
    model.add(layers.LeakyReLU(alpha= leak_alpha))
    model.add(layers.MaxPool2D(padding='same'))
    model.add(layers.Dropout( dropout_prob))

    # Second convolution layer : 3x3 filter, depth 128
    model.add(layers.Conv2D(64, 3, padding='same'))
    model.add(layers.LeakyReLU(alpha= leak_alpha))
    model.add(layers.MaxPool2D(padding='same'))
    model.add(layers.Dropout( dropout_prob))

    # Third convolution layer : 3x3 filter, depth 128
    model.add(layers.Conv2D(64, 3, padding='same'))
    model.add(layers.LeakyReLU(alpha= leak_alpha))
    model.add(layers.MaxPool2D(padding='same'))
    model.add(layers.Dropout( dropout_prob))
    model.add(layers.Flatten())

    # Fourth fully connected layer : 128 node
    model.add(layers.Dense(64, kernel_regularizer=keras.regularizers.l2( regularization_value)))
    model.add(layers.LeakyReLU(alpha= leak_alpha))
    model.add(layers.Dropout( dropout_prob * 2))

    # Softmax activation function
    model.add(
        layers.Dense( 2, kernel_regularizer=keras.regularizers.l2( regularization_value),
                     activation='sigmoid'))

    # Define the model
    return model

