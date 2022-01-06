import numpy as np
import argparse
import os
import PIL
import PIL.Image
import tensorflow as tf
import tensorflow_datasets as tfds
import pathlib
import matplotlib.pyplot as plt
from tensorflow.keras import layers
from tensorflow import keras

#dataset_url = "https://storage.googleapis.com/download.tensorflow.org/example_images/flower_photos.tgz"
#data_dir = tf.keras.utils.get_file(origin=dataset_url,
#                                   fname='flower_photos',
#                                   untar=True)

#data_dir = "../../New Plant Diseases Dataset(Augmented)/New Plant Diseases Dataset(Augmented)/train";
ap = argparse.ArgumentParser()
ap.add_argument("-d", "--dataset", required = True,
	help = "Path to the input dataset")
args = vars(ap.parse_args())

data_dir = pathlib.Path(args["dataset"])
print(data_dir)

batch_size = 32
img_height = 180
img_width = 180

dataset = tf.keras.preprocessing.image_dataset_from_directory(
    data_dir, image_size=(img_height, img_width), batch_size=batch_size
)

print("+Dataset info: ", dataset)

#train_ds = tf.keras.utils.image_dataset_from_directory(
#  data_dir,
#  validation_split=0.2,
#  subset="training",
#  seed=123,
#  image_size=(img_height, img_width),
#  batch_size=batch_size)

#val_ds = tf.keras.utils.image_dataset_from_directory(
#  data_dir,
#  validation_split=0.2,
#  subset="validation",
#  seed=123,
#  image_size=(img_height, img_width),
#  batch_size=batch_size)

class_names = np.array(sorted([item.name for item in data_dir.glob('*') if item.name != "LICENSE.txt"]))
print("+Class name info: ",class_names)

class_names = dataset.class_names
plt.figure(figsize=(20, 20))
for images, labels in dataset.take(1):
    for i in range(30):
        ax = plt.subplot(5, 6, i + 1)
        plt.imshow(images[i].numpy().astype("uint8"))
        plt.title(class_names[labels[i]])
        plt.axis("off")

plt.show()