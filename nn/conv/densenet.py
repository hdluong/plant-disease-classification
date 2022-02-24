from tensorflow.keras.applications.densenet import DenseNet201
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense, GlobalAveragePooling2D, Dropout

class DenseNet:
    @staticmethod
    def create(width = 224, height = 224, channel = 3, classes = 10):
        inputShape = (width, height, channel)
        base_model = DenseNet201(include_top=False,
                         input_shape=inputShape,
                         weights='imagenet',
                         pooling="avg")
        base_model.trainable = False
        image_input = Input(inputShape)

        x = base_model(image_input,training = False)

        x = Dense(256,activation = "relu")(x)
        x = Dropout(0.2)(x)

        x = Dense(128,activation = "relu")(x)
        x = Dropout(0.2)(x)

        image_output = Dense(10,activation="softmax")(x)
        model = Model(image_input,image_output)

        return model, base_model