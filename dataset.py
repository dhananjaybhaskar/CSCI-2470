import random
import numpy as np
import tensorflow as tf

def random_crop(img1, img2, random_crop_size):
    assert img1.shape == img2.shape
    assert img1.shape[2] == 3

    height, width = img1.shape[0], img1.shape[1]
    dy, dx = random_crop_size
    x = np.random.randint(0, width - dx + 1)
    y = np.random.randint(0, height - dy + 1)

    return img1[y:(y+dy), x:(x+dx), :], img2[y:(y+dy), x:(x+dx), :]


def crop_generator(batches, crop_size):
    while True:
        batch_x, batch_y = next(batches)

        batch_crops_x = np.zeros((batch_x.shape[0], crop_size[0], crop_size[1], 3))
        batch_crops_y = np.zeros((batch_y.shape[0], crop_size[0], crop_size[1], 3))
        for i in range(batch_x.shape[0]):
            batch_crops_x[i], batch_crops_y[i] = random_crop(batch_x[i], batch_y[i], crop_size)

        yield (batch_crops_x, batch_crops_y)

def function_generator(batches, x_function, y_function):
    while True:
        batch_x, batch_y = next(batches)

        for i in range(batch_x.shape[0]):
            batch_x[i] = x_function(batch_x[i])
            batch_y[i] = y_function(batch_x[i])

        yield (batch_x, batch_y)


def synthetic_noise_dataset(data_dir, batch_size, image_size, crop_size, noise_source_function, noise_target_function):
    generator = tf.keras.preprocessing.image.ImageDataGenerator()
    generator = generator.flow_from_directory(data_dir, image_size, batch_size=batch_size, class_mode='input')
    steps_per_epoch = len(generator)
    generator = crop_generator(generator, crop_size)
    generator = function_generator(generator, noise_source_function, noise_target_function)

    return generator, steps_per_epoch
