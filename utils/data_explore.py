import cv2
import numpy as np
import glob
import argparse
import pandas as pd
import os
import math
import tensorflow as tf

BASE_PATH = os.getcwd()
DATASETS_PATH = BASE_PATH + '/datasets/train/Tomato___Early_blight/*.jpg'

class_train_list = os.listdir(BASE_PATH + '/datasets/train')
class_valid_list = os.listdir(BASE_PATH + '/datasets/valid')

class DataExplore():
    def __init__(self, datasets_path=DATASETS_PATH, num_show=10):
        super(DataExplore, self).__init__()
        self.datasets_path = datasets_path
        self.num_show = num_show
        if self.num_show <= 0:
            self.num_show = 10
        if self.num_show > 1000:
            self.num_show = 1000

    def showClass(self):
        df = pd.DataFrame(columns=["Class","Train","Valid"])
        i = 1
        for cl in class_train_list:
            train_count = len(os.listdir(BASE_PATH + '/datasets/train/' + cl))
            if class_valid_list.__contains__(cl):
                valid_count = len(os.listdir(BASE_PATH + '/datasets/valid/' + cl))
            else:
                valid_count = 0
            df2 = pd.DataFrame([[cl, train_count, valid_count]], columns=["Class","Train","Valid"], index=[i])
            df = df.append(df2)
            i=i+1
        print(df)

    def showData(self):
        number_of_element = self.num_show
        if is_square(number_of_element):
            row = int(math.sqrt(number_of_element))
            if number_of_element < 30:
                smp = ShowMatrixPic(width=150, height=150, row=row,column=row, atuoTile=True)
            else:
                smp = ShowMatrixPic(width=100, height=100, row=row,column=row, atuoTile=True)
        elif number_of_element < 20:
            if number_of_element > 10:
                number_of_element = 10
            if number_of_element % 2 == 0:
                smp = ShowMatrixPic(width=150, height=150, row=2, column=(int(number_of_element/2)), atuoTile=True)
            else:
                smp = ShowMatrixPic(width=150, height=150, row=1,column=number_of_element, atuoTile=True)
        elif number_of_element < 30:
            smp = ShowMatrixPic(width=100, height=100, row=4,column=5, atuoTile=True)
        elif number_of_element < 40:
            smp = ShowMatrixPic(width=100, height=100, row=5,column=6, atuoTile=True)
        elif number_of_element < 50:
            smp = ShowMatrixPic(width=100, height=100, row=5,column=8, atuoTile=True)
        elif number_of_element < 100:
            smp = ShowMatrixPic(width=100, height=100, row=5,column=10, atuoTile=True)
        elif number_of_element < 500:
            smp = ShowMatrixPic(width=100, height=100, row=10,column=10, atuoTile=True)
        elif number_of_element < 1000:
            smp = ShowMatrixPic(width=50, height=50, row=20,column=25, atuoTile=True)
        elif number_of_element == 1000:
            smp = ShowMatrixPic(width=40, height=40, row=25,column=40, atuoTile=True)

        imgListOne = glob.glob(self.datasets_path)
        numpy_horizontal = smp.showPic(np.random.choice(imgListOne, number_of_element))
        cv2.imshow('img', numpy_horizontal)
        cv2.waitKey(0)

    def analysisData(self, classNm):
        imgList = glob.glob(BASE_PATH + '/datasets/train/' + classNm + '/*.jpg')
        imgSizeList = []
        for img in imgList:
            im = cv2.imread(img)
            if (not imgSizeList.__contains__(im.shape)):
                imgSizeList.append(im.shape)
        print(imgSizeList)

        # Sets up a timestamped log directory.
        logdir = "logs/train_data/"
        # Creates a file writer for the log directory.
        file_writer = tf.summary.create_file_writer(logdir)

        # Using the file writer, log the reshaped image.
        with file_writer.as_default():
            # tf.summary.image("Training data", img, step=0)
            image1 = tf.random.uniform(shape=[8, 8, 1])
            image2 = tf.random.uniform(shape=[8, 8, 1])
            tf.summary.image("grayscale_noise", [image1, image2], step=0)

class ShowMatrixPic():
    def __init__(self, row=0, column=0, atuoTile=False, width=200, height=200, text='None'):
        super(ShowMatrixPic, self).__init__()
        self.row = row
        self.column = column
        self.atuoTile = atuoTile
        self.width = width
        self.height = height
        self.text = text
        if self.row < 0:
            self.row = 0
        if self.column < 0:
            self.column = 0

        # user32 = ctypes.windll.user32
        # screensize = user32.GetSystemMetrics(0), user32.GetSystemMetrics(1)
        # print(screensize)
        #

    def __rawAndColumn(self, imgList):
        r = c = 0
        resSubtraction = 1
        if self.column == 0 and self.row == 0:
            lenList = len(imgList)
            k = round(math.sqrt(lenList))
            if k > 5:
                r = 5
                if lenList % r == 0:
                    c = lenList // k
                else:
                    num2 = lenList // r
                    if num2 > r:
                        c = k + (num2 - r)
                    else:
                        c = k + num2

            else:
                r = k
                if lenList % r == 0:
                    c = lenList // r
                else:
                    u = r ** 2
                    num2 = lenList % r
                    if u < lenList:
                        if num2 < r:
                            c = r + 1
                        else:
                            c = r + num2
                    else:
                        c = r
        elif self.column == 0 and self.row != 0:
            lenList = len(imgList)
            r = self.row
            c = math.ceil(lenList / self.row)
        elif self.column != 0 and self.row == 0:
            lenList = len(imgList)
            c = self.column
            r = math.ceil(lenList / self.column)
        else:
            r = self.row
            c = self.column

        return r, c

    def showPic(self, imgList):

        r, c = self.__rawAndColumn(imgList)

        image = []
        l = len(imgList)
        print(l)
        for i in range(l):
            img = cv2.imread(imgList[i])
            img = cv2.resize(img, (self.width, self.height),
                             interpolation=cv2.INTER_AREA)
            image.append(img)
        if not self.atuoTile:
            lenOfimg = len(imgList)
            tableNum = r * c
            emptyImg = np.zeros((self.height, self.width, 3), np.uint8)
            h, w = emptyImg.shape[:2]
            textSize = round(w * 0.005, 2)
            textThick = round(w / 100)
            cv2.putText(emptyImg, self.text, ((w // 4), (h // 2)), cv2.FONT_HERSHEY_COMPLEX,
                        textSize, (255, 255, 255), textThick)
            if (tableNum - lenOfimg) > 0:

                for o in range(tableNum - lenOfimg):
                    image.append(emptyImg)

        numpy_vertical = []

        for n in range(c):
            numpy_vertical.append(np.vstack((image[n * r:r + (n * r)])))
        numpy_horizontal = np.hstack(numpy_vertical)
        return numpy_horizontal

def is_square(n):
    sqrt = math.sqrt(n)
    return (sqrt - int(sqrt)) == 0

if __name__ == '__main__':
    # construct the argument parser and parse the arguments
    ap = argparse.ArgumentParser()
    ap.add_argument('-m', '--method', choices=['show_data', 'show_class', 'analysis_data'], help='Name of method to use.', default='show')
    ap.add_argument('-n', '--number', help='number of images to display.')
    ap.add_argument('-c', '--class', help='class name')

    # set up command line arguments conveniently
    args = vars(ap.parse_args())
    method = args['method'].upper()
    classNm = args['class']

    if args['number'] == None:
        number_of_element =  0
    else:
        number_of_element =  int(args['number'])

    de = DataExplore(DATASETS_PATH, number_of_element)

    if method == "SHOW_DATA":
        de.showData()
    elif method == "SHOW_CLASS":
        de.showClass()
    elif method == "ANALYSIS_DATA":
        de.analysisData(classNm)
