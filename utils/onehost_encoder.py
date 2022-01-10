import os
from sklearn.preprocessing import OneHotEncoder
import argparse

from data_explore import DataExplore

BASE_PATH = os.getcwd()
DATASETS_PATH = BASE_PATH + '/datasets'

class OneHotEncoderDecoder():
    def __init__(self, datasets_path=DATASETS_PATH):
        super(OneHotEncoderDecoder, self).__init__()
        self.datasets_path = datasets_path
        de = DataExplore(self.datasets_path)
        df_class = de.getClass(True)
        self.onehot = OneHotEncoder(sparse=False)
        self.onehot_encoded = self.onehot.fit_transform(df_class[["Class"]])

    def encoder(self):
        for o in self.onehot_encoded:
            print(o)
        return self.onehot_encoded

    def decoder(self, onehot_encoded):
        prediction_decoded = self.onehot.inverse_transform(onehot_encoded)
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
    labelEnCodeDecoder = OneHotEncoderDecoder(path)
    onehot_encoded = labelEnCodeDecoder.encoder()
    labelEnCodeDecoder.decoder([onehot_encoded[3]])