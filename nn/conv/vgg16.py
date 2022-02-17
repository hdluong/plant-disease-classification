from tensorflow.keras.applications.vgg16 import VGG16
from tensorflow.keras.layers import Input, Flatten, Dense, Dropout
from tensorflow.keras.models import Model
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.vgg16 import preprocess_input
import numpy as np

class VGG16_Net:
    @staticmethod
    def create(width = 224, height = 224, channel = 3, classes = 10):
        inputShape = (width, height, channel)
        X_input = Input(shape = inputShape)
        base_model = VGG16(weights = 'imagenet', include_top = False, input_tensor = X_input)

        X = base_model.output
        # FLATTEN X + FULLY_CONNECTED
        X = Flatten(name = 'flatten')(X)
        X = Dense(256, activation = 'relu', name = 'fc1')(X)
        # X = Dropout(0.5)(X)
        X = Dense(40, activation = 'relu', name = 'fc2')(X)
        X = Dense(classes, activation='softmax', name = 'predictions')(X)

        # Create Keras model instance
        model = Model(inputs = X_input, outputs = X)

        return model, base_model