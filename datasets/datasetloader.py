import numpy as np
import tensorflow as tf
from utils.onehot_encoder import OneHotEncoderDecoder
import cv2

class Dataset:
    def __init__(self, data, label, preprocessors=None):
        # the paths of images
        self.data = np.array(data)
        # the paths of segmentation images

        # binary encode
        # onehot_encoder = OneHotEncoderDecoder(label)

        self.label = label
        self.preprocessors = preprocessors
        if self.preprocessors is None:
            self.preprocessors = []

    def __len__(self):
        return len(self.data)

    def __getitem__(self, i):
        # read data
        image = cv2.imread(self.data[i])
        #image = cv2.resize(image, (self.w, self.h))
        if self.preprocessors is not None:
            for p in self.preprocessors:
                image = p.preprocess(image)
        
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