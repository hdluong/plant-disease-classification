import os
os.environ['KMP_DUPLICATE_LIB_OK']='True'
from keras.callbacks import EarlyStopping
from callbacks import *

early_stopping = EarlyStopping(
    monitor='val_loss', 
    patience=3, 
    min_delta=0.001, 
    mode='max'
)

model = create_model()
history = model.fit(
    X_train, 
    y_train, 
    epochs=50, 
    validation_split=0.20, 
    batch_size=64, 
    verbose=2,
    callbacks=[early_stopping]
)

plot_metric(history, 'loss')