# -*- coding: utf-8 -*-

from __future__ import absolute_import, division, print_function, unicode_literals

import os
import csv

import tensorflow as tf
from tensorflow import keras
import IPython.display as display
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.preprocessing.image import ImageDataGenerator


print(tf.version.VERSION)

BATCH_SIZE = 32
IMG_HEIGHT = 96
IMG_WIDTH = 96
IMG_SIZE = 96
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
                                               shuffle=True,
                                               target_size=(IMG_HEIGHT, IMG_WIDTH))
valid_data_gen = data_generator.flow_from_directory(batch_size=BATCH_SIZE,
                                               directory=valid_dir,
                                               shuffle=True,
                                               target_size=(IMG_HEIGHT, IMG_WIDTH))
test_data_gen = data_generator.flow_from_directory(batch_size=BATCH_SIZE,
                                               directory=test_dir,
                                               shuffle=True,
                                               target_size=(IMG_HEIGHT, IMG_WIDTH))


IMG_SHAPE = (IMG_SIZE, IMG_SIZE, 3)

base_model = tf.keras.applications.MobileNetV2(input_shape=IMG_SHAPE,
                                               include_top=False,
                                               weights='imagenet')

model = tf.keras.Sequential([
  base_model,
  keras.layers.GlobalAveragePooling2D(),
  keras.layers.Dense(2048, activation='relu'),
  keras.layers.Dense(6, activation="softmax")
])

base_learning_rate = 0.0001
model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=base_learning_rate),
              loss='categorical_crossentropy',
              metrics=['accuracy'])

validation_steps = 20
loss0,accuracy0 = model.evaluate(valid_data_gen, steps = validation_steps)

model.summary()

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
model.evaluate(test_data_gen)
model.save("./mobilenet")

