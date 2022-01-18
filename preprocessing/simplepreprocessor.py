import cv2

"""
Building an simple image preprocessor
ex: resize the image to a fixed size
--
"""
class SimplePreprocessor:
    
    def __init__(self, width, height):
        """
        arguments:
        width -- width of the image
        height -- height of the image
        --
        --
        """
        self.width = width
        self.height = height
    
    def preprocess(self, image):
        """
        arguments:
        image -- the source image, numpy array
        """
        image = cv2.resize(image, (self.width, self.height))
        return image