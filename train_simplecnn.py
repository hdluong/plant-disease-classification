import tensorflow as tf
import tensorflow.keras as keras
from tensorflow import keras
from nn.conv.simpleconvnet import SimpleNet
from datasets.datasetloader import Dataset, Dataloader
from preprocessing.simplepreprocessor import SimplePreprocessor
from preprocessing.hsvsegmentpreprocessor import HsvSegmentPreprocessor
from utils.onehot_encoder import OneHotEncoderDecoder
from datasets.dataset_utils import *
from callbacks.callbacks import plot_lr, plot_metric
from utils.set_logger import set_logger
from utils.set_params import Params
import pickle
import argparse
import logging

parser = argparse.ArgumentParser()
parser.add_argument('--model_dir', default='experiments/test',
                    help="Experiment directory containing params.json")
parser.add_argument('--data_dir', default='datasets/data/tomato-diseases',
                    help="Directory containing the dataset")
parser.add_argument('--restore_from', default=None,
                    help="Optional, directory or file containing weights to reload before training")
args = vars(parser.parse_args())

 # Set the logger
set_logger(os.path.join(args["model_dir"], 'train.log'))

# Define the data directories
train_path = os.path.join(args["data_dir"], 'train')
dev_path = os.path.join(args["data_dir"], 'dev')
test_path = os.path.join(args["data_dir"], 'test')

logging.info("Creating the datasets...")
X_train, y_train, X_valid, y_valid, classes = load_train_sets(train_path, dev_path)
X_test, y_test = load_test_set(test_path)

# Load the parameters from json file
json_path = os.path.join(args["model_dir"], 'params.json')
assert os.path.isfile(json_path), "No json configuration file found at {}".format(json_path)
params = Params(json_path)

height = params.image_size
width = params.image_size
channels = params.num_channels
batch_size = params.batch_size
num_epochs = params.num_epochs

# Convert training and validation labels to one hot matrices
label_encoding = OneHotEncoderDecoder(y_train)
y_train = label_encoding.onehot_encoded
y_valid = label_encoding.onehot_encoded
y_test = label_encoding.onehot_encoded

# save the OneHotEncoderDecoder to disk
label_encode_path = "utils/labels.pickle"
with open(label_encode_path, "wb") as f:
    f.write(pickle.dumps(label_encoding))
    f.close()

sp = SimplePreprocessor(width, height)
hsvSegment = HsvSegmentPreprocessor()

# Build dataset
logging.info("Build dataset...")
#train_dataset = Dataset(X_train, y_train, preprocessors=[sp, hsvSegment])
#valid_dataset = Dataset(X_valid, y_valid, preprocessors=[sp, hsvSegment])
train_dataset = Dataset(X_train, y_train)
valid_dataset = Dataset(X_valid, y_valid)

# Loader
train_loader = Dataloader(train_dataset, batch_size, len(train_dataset))
valid_loader = Dataloader(valid_dataset, batch_size, len(valid_dataset))

# model
model = SimpleNet.create(width, height, channels, classes)
model.summary()
model.compile(
	optimizer = keras.optimizers.Adam(learning_rate=params.learning_rate), 
	loss = "categorical_crossentropy",  
	metrics = ["accuracy"])

checkpoint_path = os.path.sep.join([args["model_dir"], "ckpt", "mymodel-{epoch:03d}-{val_loss:.4f}"])
checkpoint_callback = keras.callbacks.ModelCheckpoint(
	filepath=checkpoint_path,
    save_best_only=True,  # Only save a model if `val_loss` has improved.
    monitor="val_loss",
    verbose=1,)

logging.info("Starting training for {} epoch(s)".format(num_epochs))
hist = model.fit(train_loader, validation_data=valid_loader, epochs=num_epochs, callbacks=[checkpoint_callback], verbose=1)
plot_metric(hist, "loss")
#
logging.info("Evaluating network...")
results = model.evaluate(X_test, y_test, batch_size=128)
print("test loss, test acc:", results)