# Callback to save the Keras model or model weights at some frequency.
from callbacks import *
from keras.callbacks import ModelCheckpoint

def get_test_accuracy(model, x_test, y_test):
    test_loss, test_acc = model.evaluate(x=x_test, y=y_test, verbose=0)
    return test_acc

checkpoint_path = 'callbacks/model_checkpoints/'
checkpoint = ModelCheckpoint(
    filepath=checkpoint_path,
    save_freq='epoch',
    save_weights_only=True,
    verbose=1
)
# Create a new model
model = create_model()
history_checkpoint = model.fit(
    X_train, 
    y_train, 
    epochs=5, 
    validation_split=0.20, 
    batch_size=64, 
    verbose=2,
    callbacks=[checkpoint]
)

point = get_test_accuracy(model, X_test, y_test)
print('accuracy (checkpoint save): {acc:0.3f}'.format(acc=point))

# new model
new_model = create_model()
before = get_test_accuracy(new_model, X_test, y_test)
print('accuracy (Without loading weight): {acc:0.3f}'.format(acc=before))

# load weights
new_model.load_weights(checkpoint_path)
after = get_test_accuracy(new_model, X_test, y_test)
print('accuracy (Load weights): {acc:0.3f}'.format(acc=after))