from nn.conv.simpleconvnet import SimpleNet
from datasets.datasetloader import Dataset, Dataloader
from preprocessing.simplepreprocessor import SimplePreprocessor
from preprocessing.hsvsegmentpreprocessor import HsvSegmentPreprocessor
from utils.onehot_encoder import OneHotEncoderDecoder
from datasets.dataset_utils import *
from sklearn.metrics import classification_report
import pickle
import argparse

ap = argparse.ArgumentParser()
ap.add_argument("-d", "--dataset", required=True,
	help="path to input train dataset")
ap.add_argument("-m", "--model", required=True,
	help="path to output model")
ap.add_argument("-l", "--label_encode", required=True,
	help="path to output label one-hot encoder decoder")
args = vars(ap.parse_args())

#train_path = "/datasets/tomatos/train/"
train_path = args["dataset"]
height = 256
width = 256
channels = 3
batch_size = 32
num_epochs = 20

X_train, y_train, X_valid, y_valid, classes = load_train_sets(train_path)
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
train_dataset = Dataset(X_train, y_train, preprocessors=[sp, hsvSegment])
valid_dataset = Dataset(X_valid, y_valid, preprocessors=[sp, hsvSegment])

# Loader
train_loader = Dataloader(train_dataset, batch_size, len(train_dataset))
valid_loader = Dataloader(valid_dataset, batch_size, len(valid_dataset))

# model
model = SimpleNet.create(width, height, channels, classes)
model.summary()
model.compile(optimizer = "Adam", loss = "categorical_crossentropy",  metrics = ["accuracy"])

print("[INFO]: training network...")
hist = model.fit_generator(train_loader, validation_data=valid_loader, epochs=num_epochs, verbose=1)
#
#
print("[INFO] evaluating network...")
#
#