from os import name
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers
from tensorflow.keras.layers import Input, Dense, Activation, ZeroPadding2D, BatchNormalization, Flatten, Conv2D
from tensorflow.keras.layers import MaxPooling2D, Dropout
from tensorflow.keras.models import Model
from tensorflow.keras.preprocessing import image
from tensorflow.keras.utils import layer_utils
from tensorflow.keras.utils import get_file, model_to_dot, plot_model
from tensorflow.keras.applications.imagenet_utils import preprocess_input
import tensorflow.keras.backend as K

class SimpleNet:
    @staticmethod
    def create(width, height, depth, classes):
        inputShape = (width, height, depth)

        # if image data format is 'channels_first',
        # then update the input shape
        if K.image_data_format() == "channels_first":
            inputShape = (depth, width, height)
        
        X_input = Input(inputShape)

        #X = ZeroPadding2D((3,3))(X_input)

        # CONV -> BN -> RELU Block 1
        X = Conv2D(32, (3,3), strides = (1,1), name = 'conv1')(X_input)
        X = BatchNormalization(axis = 3, name = 'bn1')(X)
        X = Activation('relu')(X)
        # MAXPOOL
        X = MaxPooling2D((2,2), strides = 2, name = 'max_pool1')(X)

        # CONV -> BN -> RELU Block 2
        X = Conv2D(64, (3,3), strides = (1,1), name = 'conv2')(X)
        X = BatchNormalization(axis = 3, name = 'bn2')(X)
        X = Activation('relu')(X)
        # MAXPOOL
        X = MaxPooling2D((2,2), strides = 2, name = 'max_pool2')(X)

        # CONV -> BN -> RELU Block 3
        X = Conv2D(128, (3,3), strides = (1,1), name = 'conv3')(X)
        X = BatchNormalization(axis = 3, name = 'bn3')(X)
        X = Activation('relu')(X)
        # MAXPOOL
        X = MaxPooling2D((2,2), strides = 2, name = 'max_pool3')(X)

        # CONV -> BN -> RELU Block 4
        X = Conv2D(256, (3,3), strides = (1,1), name = 'conv4')(X)
        X = BatchNormalization(axis = 3, name = 'bn4')(X)
        X = Activation('relu')(X)
        # MAXPOOL
        X = MaxPooling2D((2,2), strides = 2, name = 'max_pool4')(X)

        # CONV -> BN -> RELU Block 5
        X = Conv2D(512, (3,3), strides = (1,1), name = 'conv5')(X)
        X = BatchNormalization(axis = 3, name = 'bn5')(X)
        X = Activation('relu')(X)
        # MAXPOOL
        X = MaxPooling2D((2,2), strides = 2, name = 'max_pool5')(X)

        # FLATTEN X + FULLY_CONNECTED
        X = Flatten()(X)
        X = Dense(40, activation = 'relu', name = 'fc6')(X)
        X = Dense(40, activation = 'relu', name = 'fc7')
        X = Dense(classes, activation = 'softmax', name = 'fc8')(X)

        # Create Keras model instance
        model = Model(inputs = X_input, outputs = X, name = 'SimpleModel')

        return model
