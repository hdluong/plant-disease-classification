import argparse
import random
import os
from tqdm import tqdm

parser = argparse.ArgumentParser()
parser.add_argument('--data_dir', default='data/tomato-diseases', help="Directory with the tomato-dieases dataset")

if __name__ == '__main__':
    args = parser.parse_args()

    assert os.path.isdir(args.data_dir), "Couldn't find the dataset at {}".format(args.data_dir)

    # Define the data directories
    train_data_dir = os.path.join(args.data_dir, 'train')
    dev_data_dir = os.path.join(args.data_dir, 'dev')

    # rename valid dir to test dir
    #valid_data_dir = os.path.join(args.data_dir, 'valid')
    #os.rename(valid_data_dir, test_data_dir)

    # Get the filenames in each directory
    filenames = []
    for folder in os.listdir(train_data_dir):
        for file in os.listdir(os.path.join(train_data_dir, folder)):
            file_path = os.path.join(train_data_dir, folder, file)
            filenames.append(file_path)
            
    # Split the images in 'train' into 80% train and 20% dev
    random.seed(230)
    filenames.sort()
    random.shuffle(filenames)

    split = int(0.8 * len(filenames))
    train_filenames = filenames[:split]
    dev_filenames = filenames[split:]

    if not os.path.exists(dev_data_dir):
        os.mkdir(dev_data_dir)
    
    for filename in tqdm(dev_filenames):
        class_dev_dir = os.path.join(dev_data_dir, filename.split('/')[-2])
        if not os.path.exists(class_dev_dir):
            os.mkdir(class_dev_dir)
        os.rename(filename, os.path.join(class_dev_dir, filename.split('/')[-1]))

    print("Done building dataset")