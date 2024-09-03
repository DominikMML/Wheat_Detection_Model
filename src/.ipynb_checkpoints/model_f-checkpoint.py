"""
Copyright 2020 Matt Bast

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
"""

import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, InputLayer, Dropout, Flatten, Reshape
from tensorflow.keras.layers import Conv2D, MaxPooling2D, GlobalMaxPooling2D
from tensorflow.keras.regularizers import l2
from tensorflow import keras
import tensorflow.keras.backend as K
import sys

def YOLO_v3(width = 256,height = 256):
    """
    Implements a YOLOv3-like convolutional neural network model.

    This function creates an uncompiled YOLOv3 model architecture for object detection.
    The model is based on the original YOLOv3 architecture by Matt Bast, with modifications
    to allow for customizable input dimensions (width and height). The model utilizes 
    convolutional layers, batch normalization, and LeakyReLU activation to construct a deep 
    neural network capable of detecting objects in an image.

    Parameters:
    - width: int, optional
        The width of the input images to the model. Default is 256 pixels.
    - height: int, optional
        The height of the input images to the model. Default is 256 pixels.

    Returns:
    - model: tf.keras.Model
        An uncompiled Keras model representing the YOLOv3 architecture for object detection.
    """
    
    x_input = tf.keras.Input(shape=(width, height, 3))

    x = tf.keras.layers.Conv2D(32, (3, 3), strides=(1, 1), padding='same')(x_input)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.LeakyReLU(alpha=0.1)(x)

    ########## block 1 ##########
    x = tf.keras.layers.Conv2D(64, (3, 3), strides=(1, 1), padding='same')(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.LeakyReLU(alpha=0.1)(x)

    x_shortcut = x

    for i in range(2):
        x = tf.keras.layers.Conv2D(32, (3, 3), strides=(1, 1), padding='same')(x)
        x = tf.keras.layers.BatchNormalization()(x)
        x = tf.keras.layers.LeakyReLU(alpha=0.1)(x)

        x = tf.keras.layers.Conv2D(64, (3, 3), strides=(1, 1), padding='same')(x)
        x = tf.keras.layers.BatchNormalization()(x)
        x = tf.keras.layers.LeakyReLU(alpha=0.1)(x)

        x = tf.keras.layers.Add()([x_shortcut, x])
        x = tf.keras.layers.LeakyReLU(alpha=0.1)(x)

        x_shortcut = x


    ########## block 2 ##########
    x = tf.keras.layers.Conv2D(128, (3, 3), strides=(1, 1), padding='same')(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.LeakyReLU(alpha=0.1)(x)

    x_shortcut = x

    for i in range(2):
        x = tf.keras.layers.Conv2D(64, (3, 3), strides=(1, 1), padding='same')(x)
        x = tf.keras.layers.BatchNormalization()(x)
        x = tf.keras.layers.LeakyReLU(alpha=0.1)(x)

        x = tf.keras.layers.Conv2D(128, (3, 3), strides=(1, 1), padding='same')(x)
        x = tf.keras.layers.BatchNormalization()(x)
        x = tf.keras.layers.LeakyReLU(alpha=0.1)(x)

        x = tf.keras.layers.Add()([x_shortcut, x])
        x = tf.keras.layers.LeakyReLU(alpha=0.1)(x)

        x_shortcut = x

    ########## block 3 ##########
    x = tf.keras.layers.Conv2D(256, (3, 3), strides=(2, 2), padding='same')(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.LeakyReLU(alpha=0.1)(x)

    x_shortcut = x

    for i in range(8):
        x = tf.keras.layers.Conv2D(128, (3, 3), strides=(1, 1), padding='same')(x)
        x = tf.keras.layers.BatchNormalization()(x)
        x = tf.keras.layers.LeakyReLU(alpha=0.1)(x)

        x = tf.keras.layers.Conv2D(256, (3, 3), strides=(1, 1), padding='same')(x)
        x = tf.keras.layers.BatchNormalization()(x)
        x = tf.keras.layers.LeakyReLU(alpha=0.1)(x)

        x = tf.keras.layers.Add()([x_shortcut, x])
        x = tf.keras.layers.LeakyReLU(alpha=0.1)(x)

        x_shortcut = x

    ########## block 4 ##########
    x = tf.keras.layers.Conv2D(512, (3, 3), strides=(2, 2), padding='same')(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.LeakyReLU(alpha=0.1)(x)

    x_shortcut = x

    for i in range(8):
        x = tf.keras.layers.Conv2D(256, (3, 3), strides=(1, 1), padding='same')(x)
        x = tf.keras.layers.BatchNormalization()(x)
        x = tf.keras.layers.LeakyReLU(alpha=0.1)(x)

        x = tf.keras.layers.Conv2D(512, (3, 3), strides=(1, 1), padding='same')(x)
        x = tf.keras.layers.BatchNormalization()(x)
        x = tf.keras.layers.LeakyReLU(alpha=0.1)(x)

        x = tf.keras.layers.Add()([x_shortcut, x])
        x = tf.keras.layers.LeakyReLU(alpha=0.1)(x)

        x_shortcut = x

    ########## block 5 ##########
    x = tf.keras.layers.Conv2D(1024, (3, 3), strides=(2, 2), padding='same')(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.LeakyReLU(alpha=0.1)(x)

    x_shortcut = x

    for i in range(4):
        x = tf.keras.layers.Conv2D(512, (3, 3), strides=(1, 1), padding='same')(x)
        x = tf.keras.layers.BatchNormalization()(x)
        x = tf.keras.layers.LeakyReLU(alpha=0.1)(x)

        x = tf.keras.layers.Conv2D(1024, (3, 3), strides=(1, 1), padding='same')(x)
        x = tf.keras.layers.BatchNormalization()(x)
        x = tf.keras.layers.LeakyReLU(alpha=0.1)(x)

        x = tf.keras.layers.Add()([x_shortcut, x])
        x = tf.keras.layers.LeakyReLU(alpha=0.1)(x)

        x_shortcut = x

    ########## output layers ##########
    x = tf.keras.layers.Conv2D(512, (3, 3), strides=(1, 1), padding='same')(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.LeakyReLU(alpha=0.1)(x)

    x = tf.keras.layers.Conv2D(256, (3, 3), strides=(1, 1), padding='same')(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.LeakyReLU(alpha=0.1)(x)

    x = tf.keras.layers.Conv2D(128, (3, 3), strides=(1, 1), padding='same')(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.LeakyReLU(alpha=0.1)(x)

    predictions = tf.keras.layers.Conv2D(10, (1, 1), strides=(1, 1), activation='sigmoid')(x)
    
    model = tf.keras.Model(inputs=x_input, outputs=predictions)
    
    return model