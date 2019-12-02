import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import sys

import tensorflow as tf
import numpy as np
import argparse
import cv2

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

    img = cv2.imread(args.data)
    x = np.array([img,])
    pred = model.predict(x)

    cv2.imwrite(args.output, pred[0])

if __name__ == '__main__':
    main()