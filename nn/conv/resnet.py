from tensorflow.keras.applications.resnet50 import ResNet50
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense, GlobalAveragePooling2D

class ResNet:
    @staticmethod
    def Create(width = 224, height = 224, channel = 3, classes = 10):
        inputShape = (width, height, channel)
        X_Input = Input(inputShape)
        base_model = ResNet50(weights='imagenet', include_top=False, input_tensor=X_Input)

        # Add global average pooling layer
        X = base_model.output
        X = GlobalAveragePooling2D()(X)

        # FULLY_CONNECTED
        X = Dense(1024, activation='relu')(X)
        # SOFTMAX_LAYER
        X = Dense(classes, activation='softmax')(X)

        # Create model
        model = Model(inputs=base_model.input, outputs=X)

        return model, base_model
