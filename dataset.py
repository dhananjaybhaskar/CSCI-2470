import random
import numpy as np
import tensorflow as tf
import os
import cv2
from tensorflow.keras.utils import Sequence

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

class RainImageGenerator(Sequence):
    def __init__(self, path_pair_file, base_dir, batch_size=4, image_size=256):

        f = open(path_pair_file, 'r')
        self.image_pair_paths = f.readlines()
        f.close()
        np.random.shuffle(self.image_pair_paths)
        self.base_dir = str(base_dir)

        self.image_num = len(self.image_pair_paths)
        self.batch_size = batch_size
        self.image_size = image_size
        self.index = 0


    def __len__(self):
        return self.image_num // self.batch_size

    def __getitem__(self, idx):
        batch_size = self.batch_size
        image_size = self.image_size
        x = np.zeros((batch_size, image_size, image_size, 3), dtype=np.uint8)
        y = np.zeros((batch_size, image_size, image_size, 3), dtype=np.uint8)
        sample_id = 0

        while True:
            image_path_pair = self.image_pair_paths[self.index]
            image_path_pair = image_path_pair.split()
            self.index += 1

            source_path = str(self.base_dir + image_path_pair[0])
            target_path = str(self.base_dir + image_path_pair[1])

            if os.path.isfile(source_path) and os.path.isfile(target_path):
                source = cv2.imread(source_path)
                target = cv2.imread(target_path)

                if (source.size != 0 and target.size != 0):
                    x[sample_id] = source
                    y[sample_id] = target
                    sample_id += 1

                if sample_id == batch_size:
                    return x, y
                

# Not used but might improve training speed if fixed
def load_pair_dataset(path_pair_file, base_dir, batch_size=4, shuffle_buffer_size=250000, n_threads=2):

    base_dir = str(base_dir)

    def load_and_process_image_pair(path_pair_file):
        path_pair = tf.strings.split(path_pair_file).values

        source = tf.io.decode_png(path_pair[0], channels=3)
        target = tf.io.decode_png(path_pair[1], channels=3)
        
        return [source, target]

    # Load dataset                                                                                                                                                                                                                                                                              
    data = np.loadtxt(path_pair_file, dtype=str)
    base_func = lambda x: [base_dir + x[0], base_dir + x[1]]
    data = np.apply_along_axis(base_func, 0, data)

    dataset = tf.data.Dataset.from_tensor_slices(data)

    # Shuffle order
    #dataset = dataset.shuffle(buffer_size=shuffle_buffer_size)

    # Load and process images (in parallel)
    dataset = dataset.map(map_func=load_and_process_image_pair, num_parallel_calls=n_threads)

    # Create batch, dropping the final one which has less than batch_size elements and finally set to reshuffle
    # the dataset at the end of each iteration
    dataset = dataset.batch(batch_size, drop_remainder=True)

    # Prefetch the next batch while the GPU is training
    dataset = dataset.prefetch(1)

    # Return an iterator over this dataset
    return dataset