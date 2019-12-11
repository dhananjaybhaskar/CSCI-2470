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
    parser.add_argument("--load", type=str, required=True,
                        help="load model weights")
    parser.add_argument("--output", type=str, required=True,
                        help="result output")
    parser.add_argument("--crop_width", type=int, default=256,
                        help="crop width")
    parser.add_argument("--crop_height", type=int, default=256,
                        help="crop height")

    args = parser.parse_args()

    return args


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

    # array to maintain all read images from the folder, same as [img,..]
    cv_img = []

    # the argument given as --data should be full path or start with ./path to go from current folder to the
    # folder than contains all the images to be read. All images will be stored in cv_image.
    # put . as --data if images are in the same folder as code.
    data_path = args.data
    data_path = data_path + "/*.png" 

    img_paths = sorted(glob.glob(data_path))

    # loop through images in the directory, add to cv_image.
    for img in img_paths:
        n = cv2.imread(img)
        n = cv2.resize(n, crop_size)
        cv_img.append(n)

    # x = np.array([img,])
    x = np.array(cv_img)

    pred = model.predict(x)


    # the argument given as --output should be full path or start with ./path from current folder to the
    # folder than contains all the images to be read. All images will be stored in cv_image.
    # put . as --data if images are in the same folder as code.
    save_path = args.output

    # loop through predictions, and name files with an index Image0, Image1, ...
    # write them to given path.
    for i,img in enumerate(pred):
        path_and_name = save_path + "/" + str(i) + ".png"
        # cv2.imwrite(args.output, pred[0])

        img = np.clip(img, 0, 255).astype(np.uint8)
        cv2.imwrite(path_and_name, img)

if __name__ == '__main__':
    main()
