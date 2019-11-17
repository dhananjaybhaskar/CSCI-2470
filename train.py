import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import sys

import tensorflow as tf
import numpy as np
import argparse

from tensorflow.keras import losses
from tensorflow.keras.callbacks import ModelCheckpoint
from pathlib import Path

from unet_model import create_unet
from dataset import synthetic_noise_dataset


def tf_log10(x):
    numerator = tf.math.log(x)
    denominator = tf.math.log(tf.constant(10, dtype=numerator.dtype))
    return numerator / denominator

def PSNR(y_true, y_pred):
    max_pixel = 255.0
    y_pred = tf.keras.backend.clip(y_pred, 0.0, 255.0)
    return 10.0 * tf_log10((max_pixel ** 2) / (tf.keras.backend.mean(tf.keras.backend.square(y_pred - y_true))))

def gaussian_noise(img, min_stddev=0, max_stddev=50):
        noise_img = img.astype(np.float)
        stddev = np.random.uniform(min_stddev, max_stddev)
        noise = np.random.randn(*img.shape) * stddev
        noise_img += noise
        noise_img = np.clip(noise_img, 0, 255).astype(np.uint8)
        return noise_img



def get_args():
    parser = argparse.ArgumentParser(description="train model",
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--data", type=str, required=True,
                        help="data location")
    parser.add_argument("--image_width", type=int, required=True,
                        help="dataset image width")
    parser.add_argument("--image_height", type=int, required=True,
                        help="dataset image height")
    parser.add_argument("--crop_width", type=int, default=256,
                        help="crop width")
    parser.add_argument("--crop_height", type=int, default=256,
                        help="crop height")
    parser.add_argument("--batch_size", type=int, default=4,
                        help="batch size")
    parser.add_argument("--lr", type=float, default=0.001,
                        help="learning rate")
    parser.add_argument("--epochs", type=float, default=1,
                        help="number of epochs")
    parser.add_argument("--load", type=str, default=None,
                        help="load model weights")
    parser.add_argument("--output_path", type=str, default="checkpoints",
                        help="checkpoint dir")
    args = parser.parse_args()

    return args


def main():
    args = get_args()
    output_path = Path(__file__).resolve().parent.joinpath(args.output_path)
    output_path.mkdir(parents=True, exist_ok=True)

    image_size = (args.image_width, args.image_height)
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



    optimizer = tf.keras.optimizers.Adam(learning_rate=args.lr, beta_1=0.9, beta_2=0.99)
    model.compile(loss='mean_squared_error', optimizer=optimizer, metrics=[PSNR])


    data_dir = Path(__file__).resolve().parent.joinpath(args.data)
    generator, steps_per_epoch = synthetic_noise_dataset(data_dir, args.batch_size, image_size, crop_size, gaussian_noise, gaussian_noise)

    checkpoint = ModelCheckpoint(str(output_path) + "/weights.{loss:.3f}-{PSNR:.5f}.hdf5",
                                 monitor="PSNR",
                                 verbose=1,
                                 mode="max",
                                 save_best_only=True)

    callbacks = [checkpoint]
    history = model.fit_generator(generator=generator,
                                  steps_per_epoch=steps_per_epoch,
                                  epochs=args.epochs,
                                  validation_data=None,
                                  verbose=1,
                                  callbacks=callbacks)


if __name__ == '__main__':
    main()