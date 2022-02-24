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
    image = Image.open(image_path).resize((image_height,image_width))
    # convert image to numpy array
    data = asarray(image)
    # summarize shape
    return data

# create dataset same as ImageDataGenerator in Keras
def create_dataset(data_dir, img_height, img_width, batch_size):
    list_ds = tf.data.Dataset.list_files(str(data_dir + '\\*\\*'), shuffle=False)
    # get the count of image files in the train directory
    image_count=0

    for dir1 in os.listdir(data_dir):
        for files in os.listdir(os.path.join(data_dir, dir1)):
            image_count+=1

    dataset = list_ds.shuffle(image_count, reshuffle_each_iteration=False)

    class_names = np.array(sorted([dir1 for dir1 in os.listdir(data_dir)]))


    #To process the label
    def get_label(file_path):
        # convert the path to a list of path components separated by sep
        parts = tf.strings.split(file_path, os.path.sep)
    
        # The second to last is the class-directory
        one_hot = parts[-2] == class_names
        # Integer encode the label
        return tf.argmax(tf.cast(one_hot, tf.int32))


    # To process the image
    def decode_img(img):
        # convert the compressed string to a 3D uint8 tensor
        img = tf.image.decode_jpeg(img, channels=3)
    
        # resize the image to the desired size
        return tf.image.resize(img, [img_height, img_width])


    # To create the single training of validation example with image and its corresponding label
    def process_path(file_path):
        label = get_label(file_path)
        # load the raw data from the file as a string
        img = tf.io.read_file(file_path)
        img = decode_img(img)
        return img, label


    AUTOTUNE = tf.data.experimental.AUTOTUNE
    # Set `num_parallel_calls` so multiple images are loaded/processed in parallel.
    dataset = dataset.map(process_path, num_parallel_calls=AUTOTUNE)


    def configure_for_performance(ds):
        ds = ds.cache()
        ds = ds.shuffle(buffer_size=1000)
        ds = ds.batch(batch_size)
        ds = ds.prefetch(buffer_size=AUTOTUNE)
        return ds


    dataset = configure_for_performance(dataset)

    return dataset
