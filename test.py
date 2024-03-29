import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import sys

import tensorflow as tf
import numpy as np
import argparse
import cv2
import glob

from tensorflow.keras import losses
from tensorflow.keras.callbacks import ModelCheckpoint, Callback, LearningRateScheduler
from tensorflow.keras.preprocessing.image import img_to_array
from pathlib import Path

from unet_model import create_unet
from dataset import synthetic_noise_dataset, RainImageGenerator

def get_args():
    parser = argparse.ArgumentParser(description="train model",
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    parser.add_argument("--data", type=str, required=True,
                        help="input data")
    parser.add_argument("--gt", type=str, required=True,
                        help="ground truth")
    parser.add_argument("--load", type=str, required=True,
                        help="load model weights")
    parser.add_argument("--crop_width", type=int, default=256,
                        help="crop width")
    parser.add_argument("--crop_height", type=int, default=256,
                        help="crop height")

    args = parser.parse_args()

    return args

def tf_log10(x):
    numerator = tf.math.log(x)
    denominator = tf.math.log(tf.constant(10, dtype=numerator.dtype))
    return numerator / denominator

def PSNR(y_true, y_pred):
    max_pixel = 255.0
    y_pred = tf.keras.backend.clip(y_pred, 0.0, 255.0)
    return 10.0 * tf_log10((max_pixel ** 2) / (tf.keras.backend.mean(tf.keras.backend.square(y_pred - y_true))))

def ssim(y_true, y_pred):
    y_pred = tf.keras.backend.clip(y_pred, 0.0, 255.0)
    return tf.reduce_mean(tf.image.ssim(y_true, y_pred, 255.0))

def main():
    args = get_args()

    #image_size = (args.image_width, args.image_height)
    crop_size = (args.crop_width, args.crop_height)


    model = create_unet(input_shape=(args.crop_width, args.crop_height, 3),
                        filters=[[48, 48], [48], [48], [48], [48], [48], [96, 96], [96, 96], [96, 96], [96, 96], [64, 32], [3]],
                        skip_start=True,
                        use_batch_norm=False,
                        use_dropout_on_upsampling=False,
                        dropout=0.0,
                        dropout_change_per_layer=0.0)

    if args.load is not None:
        model.load_weights(args.load)

    model.compile(loss='mean_squared_error', metrics=[PSNR, ssim])

    # array to maintain all read images from the folder, same as [img,..]
    cv_img = []

    # the argument given as --data should be full path or start with ./path to go from current folder to the
    # folder than contains all the images to be read. All images will be stored in cv_image.
    # put . as --data if images are in the same folder as code.
    data_path = args.data
    data_path = data_path + "/*.png"

    # loop through images in the directory, add to cv_image.
    for img in sorted(glob.glob(data_path)):
        n = cv2.imread(img)
        n = cv2.resize(n, crop_size)
        cv_img.append(n)

    # x = np.array([img,])
    x = np.array(cv_img)


    gt_img = []
    gt_path = args.gt
    gt_path = gt_path + "/*.png"

    # loop through images in the directory, add to cv_image.
    for img in sorted(glob.glob(gt_path)):
        n = cv2.imread(img)
        n = cv2.resize(n, crop_size)
        gt_img.append(n)

    y = np.array(gt_img)


    results = model.evaluate(x, y)


if __name__ == '__main__':
    main()
