from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from numpy import argmax
import argparse
import os

class OneHotEncoderDecoder():
    def __init__(self, labels):
        # integer encode
        self.label_encoder = LabelEncoder()
        integer_encoded = self.label_encoder.fit_transform(labels)
        self.integer_encoded = integer_encoded.reshape(len(integer_encoded), 1)
        # binary encode
        onehot_encoder = OneHotEncoder(sparse=False)
        self.onehot_encoded = onehot_encoder.fit_transform(self.integer_encoded)

    def encoder(self):
        for o in self.onehot_encoded:
            print(o)
        return self.onehot_encoded

    def decoder(self, preds):
        prediction = argmax(preds)
        inverted_prediction = self.label_encoder.inverse_transform([prediction])

        return inverted_prediction

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