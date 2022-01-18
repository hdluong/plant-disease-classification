from datasets.datasetloader import Dataset, Dataloader
from preprocessing.simplepreprocessor import SimplePreprocessor
from datasets.dataset_utils import *
from utils import onehot_encoder

train_path = "/datasets/tomatos/train/"
height = 100
width = 100

X_train, y_train, X_valid, y_valid, classes = load_train_sets(train_path)

sp = SimplePreprocessor(width, height)

# Build dataset
train_dataset = Dataset(X_train, y_train, preprocessors=[sp])
valid_dataset = Dataset(X_valid, y_valid, preprocessors=[sp])

# Loader
train_loader = Dataloader(train_dataset, 8, len(train_dataset))
valid_loader = Dataloader(valid_dataset, 8, len(valid_dataset))

#model = 
#hist = model.fit_generator(train_loader, validation_data=valid_loader, epochs=2, verbose=1)