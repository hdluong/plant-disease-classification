# Callback to back up and restore the training state.
from callbacks import *
from keras.callbacks import BackupAndRestore


backup_dir="backup"
backup = BackupAndRestore(backup_dir=backup_dir)

model = create_model()

history_backup = model.fit(
    X_train, 
    y_train, 
    epochs=10, 
    batch_size=1, 
    verbose=0,
    callbacks=[backup]
)