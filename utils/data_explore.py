import cv2
import numpy as np
import glob
import os
import argparse
import math

BASE_PATH = os.getcwd()
DATASETS_PATH = BASE_PATH + '/datasets/train/*.jpg'

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

    def __rawAndColumn(self,imgList):
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
            img = cv2.resize(img, (self.width, self.height), interpolation=cv2.INTER_AREA)
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
    ap.add_argument('-m', '--method', choices=['show'], help='Name of method to use.', default='show')
    ap.add_argument('-n', '--number', help='number of images to display.')
    
    # set up command line arguments conveniently
    args = vars(ap.parse_args())
    #method = METHODS[args['method'].upper()] 
    numberOfElement = int(args['number'])
    if is_square(numberOfElement):
        row = int(math.sqrt(numberOfElement));
        smp = ShowMatrixPic(width=100, height=100, row=row, column=row, atuoTile=True)
    elif numberOfElement < 20:
        if numberOfElement > 10:
            numberOfElement = 10
        if numberOfElement%2 == 0:
            smp = ShowMatrixPic(width=100, height=100, row=2, column=(int(numberOfElement/2)), atuoTile=True)
        else:
            smp = ShowMatrixPic(width=100, height=100, row=1, column=numberOfElement, atuoTile=True)
    elif numberOfElement < 30:
        numberOfElement = 20
        smp = ShowMatrixPic(width=100, height=100, row=4, column=5, atuoTile=True)
    elif numberOfElement < 40:
        smp = ShowMatrixPic(width=100, height=100, row=5, column=6, atuoTile=True)
    elif numberOfElement < 50:
        smp = ShowMatrixPic(width=100, height=100, row=5, column=8, atuoTile=True)
    elif numberOfElement < 100:
        smp = ShowMatrixPic(width=100, height=100, row=5, column=10, atuoTile=True)
    elif numberOfElement < 200:
        smp = ShowMatrixPic(width=100, height=100, row=10, column=10, atuoTile=True)
    elif numberOfElement == 500:
        smp = ShowMatrixPic(width=50, height=50, row=20, column=25, atuoTile=True)
    elif numberOfElement == 1000:
        smp = ShowMatrixPic(width=40, height=40, row=25, column=40, atuoTile=True)

    imgListOne = glob.glob(DATASETS_PATH)
    # smp = ShowMatrixPic(width=320, height=240, row=2, column=2, atuoTile=True)
    numpy_horizontal = smp.showPic(np.random.choice(imgListOne, numberOfElement))
    cv2.imshow('img', numpy_horizontal)
    cv2.waitKey(0)