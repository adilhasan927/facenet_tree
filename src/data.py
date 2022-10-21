import os
import numpy as np
# import pandas as pd
import cv2
import tensorflow as tf

# os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
# tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)

'''
Create tf.data.Dataset from a directory of Images.
Preprocess Images : Resize, Flip, Normalize
'''


def preprocess_image(image, image_size):
    
    image = tf.image.resize(image, [image_size, image_size, 16])
    # image = tf.image.random_flip_left_right(image)
    image /= 255.0
    return image


def parse_image_function(image_path, label, image_size):
      
    image_string = tf.io.read_file(image_path)
    image = tf.io.decode_raw(image_string, tf.float64)
    # image = tf.image.decode_jpeg(image_string, channels=3)
    image = preprocess_image(image, image_size)
    return image, label

# Load the numpy files
def map_func(feature_path, label, image_size):
    feature = np.load(feature_path)
    feature = feature.astype(np.float64)
    # select_idx = [0, 1, 2, 3, 4, 20, 21, 22, 23, 24]
    # feature = feature[:,:,select_idx]
    feature = np.nan_to_num(feature)
    res = cv2.resize(feature, dsize=(image_size, image_size), interpolation=cv2.INTER_CUBIC)
    return res, label

def get_dataset(dir, params, phase='train'):

    dir_paths  =  os.listdir(dir)
    dir_paths  =  [os.path.join(dir, dir_path) for dir_path in dir_paths]

    image_paths = []
    image_label = []
    for dir_path in dir_paths:
        for image_path in os.listdir(dir_path):
            image_paths.append(os.path.join(dir_path, image_path))
            image_label.append(dir_path.split('/')[-1])

    AUTOTUNE   =  tf.data.experimental.AUTOTUNE
    dataset    =  tf.data.Dataset.from_tensor_slices((image_paths, image_label))
    # dataset    =  dataset.map(lambda x, y: parse_image_function(x, y, params.image_size))
    dataset    =  dataset.map(lambda item, label: tf.numpy_function(
                              map_func, [item, label, params.image_size], [tf.float64, tf.string]))
    dataset    =  dataset.batch(params.batch_size).prefetch(AUTOTUNE)
    
    return dataset, len(image_paths)


if __name__ == "__main__":

    print(get_dataset('../face-data'))
