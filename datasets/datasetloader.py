import numpy as np
import tensorflow as tf
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
import cv2

"""
Load and preprocess the dataset from disk
--
--
"""
class DatasetLoader:
    """
    arguments:
    preprocessors -- a list contain the preprocess methods
    """
    def __init__(self, preprocessors = None):
        pass

    """
    Load dataset from disk
    arguments:
    imagePaths -- a list of file paths to images in the datasets on disk
    flagInfo -- a flag to specify to print updates to a console or not
    """
    def load(self, imagePaths, flagInfo = -1):
        pass

class Dataset:
    def __init__(self, data, label, w, h):
        # the paths of images
        self.data = np.array(data)
        # the paths of segmentation images

        # binary encode
        label_encoder = LabelEncoder()
        integer_encoded = label_encoder.fit_transform(label)


        onehot_encoder = OneHotEncoder(sparse=False)
        integer_encoded = integer_encoded.reshape(len(integer_encoded), 1)
        onehot_encoded = onehot_encoder.fit_transform(integer_encoded)


        self.label = onehot_encoded
        self.w = w
        self.h = h

    def __len__(self):
        return len(self.data)

    def __getitem__(self, i):
        print("Build model")
        # read data
        image = cv2.imread(self.data[i])
        image = cv2.resize(image, (self.w, self.h))
        label = self.label[i]
        return image, label

class Dataloader(tf.keras.utils.Sequence):
    def __init__(self, dataset, batch_size, size):
        self.dataset = dataset
        self.batch_size = batch_size
        self.size = size

    def __getitem__(self, i):
        # collect batch data
        start = i * self.batch_size
        stop = (i + 1) * self.batch_size
        data = []
        for j in range(start, stop):
            data.append(self.dataset[j])

        batch = [np.stack(samples, axis=0) for samples in zip(*data)]
        return tuple(batch)

    def __len__(self):
        return self.size // self.batch_size