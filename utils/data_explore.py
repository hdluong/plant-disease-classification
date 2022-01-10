import cv2
import numpy as np
import glob
import argparse
import pandas as pd
import os
import math
import matplotlib.pyplot as plt
from random import randint

BASE_PATH = os.getcwd()
DATASETS_PATH = BASE_PATH + '/datasets'

class DataExplore():
    def __init__(self, datasets_path=DATASETS_PATH, num_show=10):
        super(DataExplore, self).__init__()
        self.datasets_path = datasets_path
        self.num_show = num_show
        self.class_train_list = os.listdir(datasets_path + '/train')
        self.class_valid_list = os.listdir(datasets_path + '/valid')
        if self.num_show <= 0:
            self.num_show = 10
        if self.num_show > 1000:
            self.num_show = 1000

    def getClass(self, isPrint=True):
        df = pd.DataFrame(columns=["Class","Train","Valid"])
        i = 1
        for cl in self.class_train_list:
            train_count = len(os.listdir(self.datasets_path + '/train/' + cl))
            if self.class_valid_list.__contains__(cl):
                valid_count = len(os.listdir(self.datasets_path + '/valid/' + cl))
            else:
                valid_count = 0
            df2 = pd.DataFrame([[cl, train_count, valid_count]], columns=["Class","Train","Valid"], index=[i])
            df = df.append(df2)
            i=i+1
        if (isPrint): print(df)
        return df
    
    def showData(self):
        train_images = []
        train_labels = []
        test_images = []
        test_labels = []

        for train_test_folder in os.listdir(self.datasets_path):
            # if train_test_folder == 'train':
            #     train_path = os.path.join(dataset_path, train_test_folder)
            #     for disease_folder in os.listdir(train_path):
            #         train_labels.append(disease_folder)
            #         disease_path = os.path.join(train_path, disease_folder)
            #         for file in os.listdir(disease_path):
            #             if file.endswith('jpg') or file.endswith('JPG'):
            #                 img_path = os.path.join(disease_path, file)
            #                 img = cv2.imread(img_path)
            #                 r, g, b = img[:, :, 0]/255, img[:, :, 1]/255, img[:, :, 2]/255
            #                 img = np.dstack((r, g, b))
            #                 train_images.append(img)
      
            if train_test_folder == 'valid':
                test_path = os.path.join(self.datasets_path, train_test_folder)
                for disease_folder in os.listdir(test_path):
                    disease_path = os.path.join(test_path, disease_folder)
                    i = 0
                    for file in os.listdir(disease_path):
                        if i < 20 and (file.endswith('jpg') or file.endswith('JPG')):
                            img_path = os.path.join(disease_path, file)
                            img = cv2.imread(img_path)
                            r, g, b = img[:, :, 0]/255, img[:, :, 1]/255, img[:, :, 2]/255
                            img = np.dstack((r, g, b))
                            test_images.append(img)
                            test_labels.append(disease_folder)
                            i=i+1
                    
        train_images = np.array(train_images)
        train_labels = np.array(train_labels)
        test_images = np.array(test_images)
        test_labels = np.array(test_labels)
        print('Shape of the stacked train images:', train_images.shape)
        print('Shape of the train labels:', train_labels.shape)
        print('Shape of the stacked test images:', test_images.shape)
        print('Shape of the test_labels:', test_labels.shape)
        unique_labels = np.unique(test_labels)
        print(unique_labels)
        row = 5
        col = 4
        fig, axes = plt.subplots(row, col, figsize=(14, 14))
        for i in range(row):
            for j in range(col):
                c = randint(0, len(test_images) - 1)
                axes[i][j].imshow(test_images[c])
                axes[i][j].set_title(test_labels[c])
        plt.tight_layout()
        plt.show()

if __name__ == '__main__':
    # construct the argument parser and parse the arguments
    ap = argparse.ArgumentParser()
    ap.add_argument('-m', '--method', choices=['show_data', 'show_class', 'analysis_data'], help='Name of method to use.', default='show')
    ap.add_argument('-n', '--number', help='number of images to display.')
    ap.add_argument('-c', '--class', help='class name')
    ap.add_argument('-p', '--path', help='datasets path')

    # set up command line arguments conveniently
    args = vars(ap.parse_args())
    method = args['method'].upper()
    classNm = args['class']
    datasets_path = args['path']

    if datasets_path:
        if not os.path.exists(datasets_path):
            print(datasets_path, ': does not exist') 
            exit()
        elif not os.path.isdir(datasets_path):
            print(datasets_path, ': is not a directory') 
            exit()
    else:
        print('Please enter your datasets path') 
        exit()

    if args['number'] == None:
        number_of_element =  0
    else:
        number_of_element =  int(args['number'])

    de = DataExplore(datasets_path, number_of_element)

    if method == "SHOW_DATA":
        de.showData()
    elif method == "SHOW_CLASS":
        de.getClass()
