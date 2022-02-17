# Reduce learning rate when a metric has stopped improving.
from keras.callbacks import ReduceLROnPlateau
from callbacks import *

reduce_lr = ReduceLROnPlateau(
    monitor='val_loss', 
    factor=0.2,   
    patience=2, 
    min_lr=0.001,
    verbose=2
)

model = create_model()

history_reduce_lr = model.fit(
    X_train, 
    y_train, 
    epochs=50, 
    validation_split=0.20, 
    batch_size=64, 
    verbose=2,
    callbacks=[reduce_lr]
)

plot_lr(history_reduce_lr)
plot_metric(history_reduce_lr, 'loss')