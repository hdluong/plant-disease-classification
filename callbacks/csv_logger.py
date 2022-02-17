# Callback that streams epoch results to a CSV file.
from keras.callbacks import CSVLogger
import time
import os
from callbacks import *

PATH = "callbacks\logs"

timestr = time.strftime("%Y%m%d-%H%M%S")
if not os.path.exists(PATH):
    os.makedirs(PATH)
    csv_log = CSVLogger(f"{PATH}\{timestr}.csv")

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