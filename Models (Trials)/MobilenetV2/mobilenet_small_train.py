# -*- coding: utf-8 -*-

from __future__ import absolute_import, division, print_function, unicode_literals

import os
import csv

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import IPython.display as display
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.preprocessing.image import ImageDataGenerator

print(tf.version.VERSION)


BATCH_SIZE = 128

class InvertedBottleNeck(keras.layers.Layer):
  def __init__(self, output_channels, input_channels, stride=1, expand=6, **kwargs):
    super(InvertedBottleNeck, self).__init__(**kwargs)
    self.add = keras.layers.Add()
    self.output_channels = output_channels
    self.input_channels = input_channels
    self.stride = stride
    self.conv1 = layers.Conv2D(kernel_size=1, 
                                strides=1, 
                                activation=None, 
                                padding="same",
                                filters=expand * input_channels)
    self.bn1 = layers.BatchNormalization()
    self.relu1 = layers.ReLU(6.)
    self.dwconv = layers.DepthwiseConv2D(kernel_size=3, 
                                strides=stride, 
                                padding="same",
                                activation=None)
    self.bn2 = layers.BatchNormalization()
    self.relu2 = layers.ReLU(6.)
    self.conv2 = layers.Conv2D(filters=output_channels, 
                                kernel_size=1, 
                                strides=1, 
                                activation=None, 
                                padding="same")
    self.bn3 = layers.BatchNormalization()
  
  def get_config(self):

    config = super().get_config().copy()
    config.update({
        'output_channels': self.output_channels,
        'input_channels': self.input_channels,
        'stride': self.stride,
    })
    return config

  def call(self, input_tensor, training=False):
    x = self.conv1(input_tensor)
    x = self.bn1(x, training=training)
    x = self.relu1(x)
    x = self.dwconv(x)
    x = self.bn2(x, training=training)
    x = self.relu2(x)
    x = self.conv2(x)
    x = self.bn3(x, training=training)
    if (self.input_channels == self.output_channels) and (self.stride == 1):
      x = self.add([x, input_tensor])

    return x

input_tensor = tf.ones((32, 64, 64, 1))
model = keras.Sequential(
      [
        layers.Conv2D(filters=48, kernel_size=3, padding='same', activation=None),
        layers.BatchNormalization(),
        layers.ReLU(6.),
        layers.Conv2D(filters=32, kernel_size=3, padding='same', activation=None),
        layers.BatchNormalization(),
        layers.ReLU(6.),
        InvertedBottleNeck(output_channels=32, input_channels=32, stride=2),
        InvertedBottleNeck(output_channels=24, input_channels=32, stride=2),
        InvertedBottleNeck(output_channels=24, input_channels=24, stride=1),
        InvertedBottleNeck(output_channels=32, input_channels=24, stride=2),
        InvertedBottleNeck(output_channels=32, input_channels=32, stride=1),
        InvertedBottleNeck(output_channels=32, input_channels=32, stride=1),
        InvertedBottleNeck(output_channels=64, input_channels=32, stride=1),
        InvertedBottleNeck(output_channels=64, input_channels=64, stride=1),
        InvertedBottleNeck(output_channels=64, input_channels=64, stride=1),
        InvertedBottleNeck(output_channels=64, input_channels=64, stride=1),
        InvertedBottleNeck(output_channels=128, input_channels=64, stride=1),
        InvertedBottleNeck(output_channels=256, input_channels=128, stride=1),
        layers.GlobalAveragePooling2D(),
        layers.Dense(units=6, activation="softmax")
      ]
  )
for layer in model.layers:
  layer.trainable = True

model(input_tensor)
model.summary()

IMG_HEIGHT = 64
IMG_WIDTH = 64
IMG_SIZE = 64 

train_dir = "./train"
test_dir = "./test"
valid_dir = "./valid"
CLASS_NAMES = np.array(['anger', 'fear', 'happiness', 'neutral', 'sadness', 'surprise'])

data_generator = ImageDataGenerator(rescale = 1./255.,
                                   rotation_range = 40,
                                   width_shift_range = 0.2,
                                   height_shift_range = 0.2,
                                   shear_range = 0.2,
                                   zoom_range = 0.2,
                                   horizontal_flip = True)

train_data_gen = data_generator.flow_from_directory(batch_size=BATCH_SIZE,
                                               directory=train_dir,
                                               color_mode='grayscale',
                                               shuffle=True,
                                               target_size=(IMG_HEIGHT, IMG_WIDTH))
valid_data_gen = data_generator.flow_from_directory(batch_size=BATCH_SIZE,
                                               directory=valid_dir,
                                               color_mode='grayscale',
                                               shuffle=True,
                                               target_size=(IMG_HEIGHT, IMG_WIDTH))
test_data_gen = data_generator.flow_from_directory(batch_size=BATCH_SIZE,
                                               directory=test_dir,
                                               color_mode='grayscale',
                                               shuffle=True,
                                               target_size=(IMG_HEIGHT, IMG_WIDTH))

base_learning_rate = 0.0001
model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=base_learning_rate),
              loss='categorical_crossentropy',
              metrics=['accuracy'])

validation_steps = 20
loss0,accuracy0 = model.evaluate(valid_data_gen, steps = validation_steps)

num_train = 28026
num_test = 3497
num_valid = 3497

initial_epochs = 2
steps_per_epoch = round(num_train)//BATCH_SIZE
validation_steps = round(num_valid)//BATCH_SIZE
history = model.fit(train_data_gen,
                    steps_per_epoch=steps_per_epoch,
                    epochs=initial_epochs,
                    validation_data=valid_data_gen,
                    validation_steps=validation_steps)
model.save("./mobilenet_small")
model.evaluate(test_data_gen)

