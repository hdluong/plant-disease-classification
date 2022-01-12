from preprocessing_data import *
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Dense, Flatten, Dropout
from keras.preprocessing.image import ImageDataGenerator

# Initial values
src_path_train = "dataset/train/"
src_path_test = "dataset/test/"
epochs = 3

height = 100
width = 100

batch_size = 32
seed = 50

# Create model
model = prepare_model(height, width)

# Preprocessing data
train_generator, valid_generator, test_generator = preprocessing(src_path_train, src_path_test, height, width, batch_size, seed)

n_train_steps = train_generator.n//train_generator.batch_size
n_valid_steps = valid_generator.n//valid_generator.batch_size

model.fit(train_generator,
            steps_per_epoch=n_train_steps,
            validation_data=valid_generator,
            validation_steps=n_valid_steps,
            epochs=epochs
)

score = model.evaluate(valid_generator)
print('Test loss:', score[0])
print('Test accuracy:', score[1])

predict=model.predict(test_generator)
# predict the class label
y_classes = predict.argmax(axis=-1)
print("Class name: ", y_classes)