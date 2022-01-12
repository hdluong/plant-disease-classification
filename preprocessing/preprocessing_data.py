from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Dense, Flatten, Dropout
from keras.preprocessing.image import ImageDataGenerator

def prepare_model(height, width):
    model = Sequential()
    model.add(Conv2D(32,kernel_size=(3,3),activation='relu',input_shape=(height, width, 3)))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Flatten())
    model.add(Dense(16, activation='relu'))
    model.add(Dense(2, activation='sigmoid'))
    model.compile(loss="binary_crossentropy",optimizer="adam",metrics=['accuracy'])
    return model

def data_generator_type_1():
    train_datagen = ImageDataGenerator(
        rescale=1 / 255.0,
        rotation_range=20,
        zoom_range=0.05,
        width_shift_range=0.05,
        height_shift_range=0.05,
        shear_range=0.05,
        horizontal_flip=True,
        fill_mode="nearest",
        validation_split=0.20)
    return train_datagen

def preprocessing(src_path_train, src_path_test, height, width, batch_size, seed):
    train_datagen = data_generator_type_1()

    test_datagen = ImageDataGenerator(rescale=1 / 255.0)

    height = 100
    width = 100

    # Training data
    train_generator = train_datagen.flow_from_directory(
        directory=src_path_train,
        target_size=(height, width),
        color_mode="rgb",
        batch_size=batch_size,
        class_mode="categorical",
        subset='training',
        shuffle=True,
        seed=seed
    )

    # Validation data
    valid_generator = train_datagen.flow_from_directory(
        directory=src_path_train,
        target_size=(height, width),
        color_mode="rgb",
        batch_size=batch_size,
        class_mode="categorical",
        subset='validation',
        shuffle=True,
        seed=seed
    )

    # Testing data
    test_generator = test_datagen.flow_from_directory(
        directory=src_path_test,
        target_size=(height, width),
        color_mode="rgb",
        batch_size=1,
        class_mode=None,
        shuffle=False,
        seed=seed
    )
    
    return train_generator, valid_generator, test_generator