from utils.segment_utils import *
import cv2

"""
Building a segment preprocessor
segment leaf image based on hsv color
"""
class HsvSegmentPreprocessor:
    
    def __init__(self):
        pass
    
    def preprocess(self, image):
        """
        arguments:
        image -- the source image, numpy array
        """
        return segment_hsv(image)