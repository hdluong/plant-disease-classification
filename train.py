"""Train the model"""

import argparse
import imp
import logging
import os
import random

import tensorflow as tf

from datasets.preprocessing_data import *
from nn.conv.simpleconvnet import SimpleNet
from datasets.datasetloader import Dataset, Dataloader
from preprocessing.simplepreprocessor import SimplePreprocessor
from preprocessing.hsvsegmentpreprocessor import HsvSegmentPreprocessor
from utils.onehot_encoder import OneHotEncoderDecoder
from datasets.dataset_utils import *
from utils.utils import *

parser = argparse.ArgumentParser()
parser.add_argument('--model_dir', default='experiments/test',
                    help="Experiment directory containing params.json")
parser.add_argument('--data_dir', default='datasets/data/tomato-diseases',
                    help="Directory containing the dataset")
parser.add_argument('--restore_from', default=None,
                    help="Optional, directory or file containing weights to reload before training")

if __name__ == '__main__':
    # Set the random seed for the whole graph for reproductible experiments
    tf.set_random_seed(230)

    # Load the parameters from json file
    args = parser.parse_args()
    json_path = os.path.join(args.model_dir, 'params.json')
    assert os.path.isfile(
        json_path), "No json configuration file found at {}".format(json_path)
    params = Params(json_path)

    # Set the logger
    set_logger(os.path.join(args.model_dir, 'train.log'))

    # Create the input data pipeline
    logging.info("Creating the datasets...")
    data_dir = args.data_dir
    train_data_dir = os.path.join(data_dir, "train")
    dev_data_dir = os.path.join(data_dir, "dev")

    # Preprocessing data
    train_generator, valid_generator, test_generator = preprocessing(train_data_dir, dev_data_dir, params.height, params.width, params.batch_size, 230)

    n_train_steps = train_generator.n//train_generator.batch_size
    n_valid_steps = valid_generator.n//valid_generator.batch_size

    # Define the model
    logging.info("Creating the model...")
    model = SimpleNet.create(params.width, params.height)

    model.fit(train_generator,
                steps_per_epoch=n_train_steps,
                validation_data=valid_generator,
                validation_steps=n_valid_steps,
                epochs=params.num_epochs
    )

    score = model.evaluate(valid_generator)
    print('Test loss:', score[0])
    print('Test accuracy:', score[1])

    predict=model.predict(test_generator)
    # predict the class label
    y_classes = predict.argmax(axis=-1)
    print("Class name: ", y_classes)