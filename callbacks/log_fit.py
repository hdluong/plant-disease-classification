# Inspecting and understanding TensorFlow runs and graphs.
from keras.callbacks import TensorBoard
from callbacks import *
import time

log_dir = "logs/fit/" + time.strftime("%Y%m%d-%H%M%S")
tensorboard_callbacks = TensorBoard(log_dir=log_dir, histogram_freq=1)

model = create_model()

model.fit(
    x=X_train, 
    y=y_train, 
    epochs=5, 
    validation_data=(X_test, y_test), 
    callbacks=[tensorboard_callbacks]
)