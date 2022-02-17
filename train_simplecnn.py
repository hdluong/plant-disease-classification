from tensorflow import keras
from sklearn.metrics import classification_report
import pickle
import argparse
import logging

from nn.conv.simpleconvnet import SimpleNet
from datasets.datasetloader import Dataset, Dataloader
from preprocessing.simplepreprocessor import SimplePreprocessor
from preprocessing.hsvsegmentpreprocessor import HsvSegmentPreprocessor
from utils.onehot_encoder import OneHotEncoderDecoder
from datasets.dataset_utils import *
from utils.set_logger import set_logger
from utils.set_params import Params

ap = argparse.ArgumentParser()
ap.add_argument("-d", "--dataset", required=True,
	help="path to input train dataset")
ap.add_argument("-m", "--model", required=True,
	help="path to output model")
ap.add_argument("-l", "--label_encode", required=True,
	help="path to output label one-hot encoder decoder")
args = vars(ap.parse_args())

# Load the parameters from json file
json_path = os.path.join(args["model"], 'params.json')
assert os.path.isfile(json_path), "No json configuration file found at {}".format(json_path)
params = Params(json_path)

#train_path = "/datasets/tomatos/train/"
train_path = json_path = os.path.join(args["dataset"], 'train')
valid_path = json_path = os.path.join(args["dataset"], 'valid')
# height = 256
# width = 256
# channels = 3
# batch_size = 32
# num_epochs = 20
height = params.image_size
width = params.image_size
channels = params.num_channels
batch_size = params.batch_size
num_epochs = params.num_epochs

set_logger('train_simplecnn.log')

X_train, y_train, X_valid, y_valid, classes = load_train_sets(train_path, valid_path)
# Convert training and validation labels to one hot matrices
label_encoding = OneHotEncoderDecoder(y_train)
y_train = label_encoding.onehot_encoded
y_valid = OneHotEncoderDecoder(y_valid).onehot_encoded

# save the OneHotEncoderDecoder to disk
with open(args["label_encode"], "wb") as f:
    f.write(pickle.dumps(label_encoding))
    f.close()

sp = SimplePreprocessor(width, height)
hsvSegment = HsvSegmentPreprocessor()

# Build dataset
logging.info("Build dataset...")
train_dataset = Dataset(X_train, y_train, preprocessors=[sp, hsvSegment])
valid_dataset = Dataset(X_valid, y_valid, preprocessors=[sp, hsvSegment])

# Loader
train_loader = Dataloader(train_dataset, batch_size, len(train_dataset))
valid_loader = Dataloader(valid_dataset, batch_size, len(valid_dataset))

# model
model = SimpleNet.create(width, height, channels, classes)
model.summary()
# model.compile(optimizer = "Adam", loss = "categorical_crossentropy",  metrics = ["accuracy"])
model.compile(keras.optimizers.Adam(learning_rate=params.learning_rate), loss = "categorical_crossentropy",  metrics = ["accuracy"])

# print("[INFO]: training network...")
logging.info("training network...")
hist = model.fit(train_loader, validation_data=valid_loader, epochs=num_epochs, verbose=1)
#
#
# print("[INFO] evaluating network...")
logging.info("evaluating network...")
#
#