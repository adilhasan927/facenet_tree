import os
import numpy as np
# import pandas as pd
import cv2
from PIL import Image
import torch
import torchvision

# os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
# tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)

'''
Create tf.data.Dataset from a directory of Images.
Preprocess Images : Resize, Flip, Normalize
'''

# Load the numpy files
def map_func(feature_path, label, image_size):
    feature = np.load(feature_path)
    feature = feature.astype(np.float64)
    # select_idx = [0, 1, 2, 3, 4, 20, 21, 22, 23, 24]
    # feature = feature[:,:,select_idx]
    return res, label

class OurDataset(torch.utils.data.Dataset):
    def __init__(image_paths, image_labels, image_size):
        self.image_paths = image_paths
        self.image_labels = image_labels
        self.image_size = image_size

    def __len__(self):
        return len(self.image_labels)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        image = np.load(img_path)
        image = image.astype('float32')
        image = np.nan_to_num(image)
        image = cv2.resize(
            image,
            dsize=(self.image_size, self.image_size),
            interpolation=cv2.INTER_CUBIC)
        image = torch.Tensor(image, dtype=torch.float64)
        label = self.image_labels[idx]
        return image, label

def get_dataset(dir, params, phase='train'):

    dir_paths  =  os.listdir(dir)
    dir_paths  =  [os.path.join(dir, dir_path) for dir_path in dir_paths]

    image_paths = []
    image_labels = []
    for dir_path in dir_paths:
        for image_path in os.listdir(dir_path):
            image_paths.append(os.path.join(dir_path, image_path))
            image_labels.append(dir_path.split('/')[-1])

    dataset =  OurDataset(image_paths, image_label)

    return dataset

def get_dataloader(dir, params, phase='train'):
    dataset = get_dataset(dir, params, phase)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=params.batch_size)
    return dataloader


if __name__ == "__main__":

    print(get_dataloader('../face-data'))
