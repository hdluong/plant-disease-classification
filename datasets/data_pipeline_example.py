import numpy as np
import os
import tensorflow as tf
from data_pipeline_function import *

data_dir=r'dataset/train'
val_dir=r'dataset/valid'
batch_size = 32
img_height = 200
img_width = 200
epochs = 3


train_ds = create_dataset(data_dir, img_height, img_width, batch_size)
val_ds = create_dataset(val_dir, img_height, img_width, batch_size)


# tf model
model=tf.keras.Sequential(
        [
            tf.keras.layers.InputLayer(input_shape=(img_height, img_width, 3)),
            tf.keras.layers.Conv2D(filters=32, kernel_size=3, strides=(2, 2), activation='relu'),
            tf.keras.layers.Conv2D(filters=64, kernel_size=3, strides=(2, 2), activation='relu'),
            tf.keras.layers.Flatten(),
            tf.keras.layers.Dense(6)
         ])

#Compile the model
model.compile(optimizer='rmsprop', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

#Fitting the model
history = model.fit(train_ds, validation_data=val_ds, epochs=epochs)
