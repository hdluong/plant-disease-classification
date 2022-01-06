import os
import pandas as pd
from sklearn.preprocessing import OneHotEncoder
import argparse

from data_explore import DataExplore

BASE_PATH = os.getcwd()
DATASETS_PATH = BASE_PATH + '/datasets/train/Tomato___Early_blight/*.jpg'

def encoder(datasets_path):
    de = DataExplore(datasets_path, 10)
    onehot = OneHotEncoder(sparse=False)
    df_train = de.showClass(True)
    onehot_encoded = onehot.fit_transform(df_train[["Class"]])
    for o in onehot_encoded:
        print(o)
    return onehot, onehot_encoded

def decoder(onehot, onehot_encoded):
    prediction_decoded = onehot.inverse_transform(onehot_encoded)
    print(prediction_decoded)
    return prediction_decoded

if __name__ == '__main__':
    # construct the argument parser and parse the arguments
    ap = argparse.ArgumentParser()
    ap.add_argument('-p', '--path', help='Datasets Path')

    # set up command line arguments conveniently
    args = vars(ap.parse_args())
    path = args['path']

    if path:
        if not os.path.exists(path):
            print(path, ': does not exist') 
            exit()
        elif not os.path.isdir(path):
            print(path, ': is not a directory') 
            exit()
    else:
        print('Please enter your datasets path') 
        exit()

    # test
    onehot, onehot_encoded = encoder(path)
    decoder(onehot, [onehot_encoded[4]])