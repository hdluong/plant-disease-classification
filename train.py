"""Train the model"""

import argparse
import logging
import os
import tensorflow as tf
import tensorflow.keras as keras

from datasets.preprocessing_data import *
from nn.conv.densenet import DenseNet
from datasets.datasetloader import Dataset, Dataloader
from preprocessing.simplepreprocessor import SimplePreprocessor
from preprocessing.hsvsegmentpreprocessor import HsvSegmentPreprocessor
from utils.onehot_encoder import OneHotEncoderDecoder
from datasets.dataset_utils import *
from utils.set_logger import set_logger
from utils.set_params import Params

parser = argparse.ArgumentParser()
parser.add_argument('--model_dir', default='experiments/test',
                    help="Experiment directory containing params.json")
parser.add_argument('--data_dir', default='datasets/data/tomato-diseases',
                    help="Directory containing the dataset")
parser.add_argument('--restore_from', default=None,
                    help="Optional, directory or file containing weights to reload before training")

if __name__ == '__main__':
    # Load the parameters from json file
    args = parser.parse_args()
    json_path = os.path.join(args.model_dir, 'params.json')
    assert os.path.isfile(json_path), "No json configuration file found at {}".format(json_path)
    params = Params(json_path)

    # Set the logger
    set_logger(os.path.join(args.model_dir, 'train.log'))

    # Create the input data pipeline
    logging.info("Creating the datasets...")
    data_dir = args.data_dir
    train_data_dir = os.path.join(data_dir, "train")
    dev_data_dir = os.path.join(data_dir, "dev")
    test_data_dir = os.path.join(data_dir, "test")

    # Preprocessing data
    logging.info("Preprocessing data")
    train_generator, valid_generator, test_generator = preprocessing(train_data_dir, dev_data_dir, test_data_dir, params.image_size, params.image_size, params.batch_size, 230)

    n_train_steps = train_generator.n//train_generator.batch_size
    n_valid_steps = valid_generator.n//valid_generator.batch_size

    # Reload model from directory if specified
    if args["restore_from"] is not None:
        logging.info("Restoring parameters from {}".format(args["restore_from"]))
        if os.path.isdir(args["restore_from"]):
            model = keras.models.load_model(args["restore_from"])
    else:
        # Define the model
        logging.info("Creating the model...")
        model, _ = DenseNet.create(params.image_size, params.image_size)

    model.summary()
    model.compile(
        optimizer = keras.optimizers.Adam(learning_rate=params.learning_rate), 
        loss = "categorical_crossentropy", metrics = ["accuracy"])
    
    checkpoint_path = os.path.sep.join([args["model_dir"], "ckpt", "mymodel-{epoch:03d}-{val_loss:.4f}"])
    checkpoint_callback = keras.callbacks.ModelCheckpoint(
        filepath=checkpoint_path,
        save_best_only=True,
        monitor="val_loss",
        verbose=1,)

    logging.info("Starting training for {} epoch(s)".format(params.num_epochs))

    model.fit(train_generator,
            steps_per_epoch=n_train_steps,
            validation_data=valid_generator,
            validation_steps=n_valid_steps,
            epochs=params.num_epochs,
            callbacks = [checkpoint_callback],
            verbose = 1)

    score = model.evaluate(valid_generator)
    print('Test loss:', score[0])
    print('Test accuracy:', score[1])

    # predict=model.predict(test_generator)
    # # predict the class label
    # y_classes = predict.argmax(axis=-1)
    # print("Class name: ", y_classes)