import numpy as np
import os
import tensorflow as tf
import pathlib
import pandas as pd
import PIL

from PIL import Image
from numpy import asarray

# prepare input for tf.data.dataset
def make_image_and_label_list(folder_dir, file_type):
    folder = folder_dir
    sub_folders = [name for name in os.listdir(folder) if os.path.isdir(os.path.join(folder, name))]
    
    images_list = []
    label = []
    
    for fd in sub_folders:
        parent_dir = folder + '/' + fd
        files = os.listdir(parent_dir)
        for f in files:
            if (f.endswith("." + file_type)):
                images_list.append(parent_dir + '/' + f)
                label.append(fd)
    
    return images_list, label

# convert image to numpy array
def convert_image_to_numpy(image_path, image_height, image_width):
    # load the image
    image = Image.open(image_path).resize((200,200))
    # convert image to numpy array
    data = asarray(image)
    # summarize shape
    return data