from ntpath import join
from keras.callbacks import CSVLogger
import time
from callbacks import *

timestr = time.strftime("%Y%m%d-%H%M%S")
csv_log = CSVLogger(f"callbacks/logs/test-{timestr}.csv")

model = create_model()
history_csv_logger = model.fit(
    X_train, 
    y_train, 
    epochs=10, 
    validation_split=0.20, 
    batch_size=64, 
    verbose=2,
    callbacks=[csv_log]
)