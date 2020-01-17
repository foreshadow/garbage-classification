import os

import numpy as np
import tensorflow as tf

AUTOTUNE = tf.data.experimental.AUTOTUNE

label1 = sum([[i] * c for i, c, in enumerate([6, 8, 23, 3])], [])


def preprocess_image(path, is_test=False):
    blob = tf.io.read_file(path)
    image = tf.image.decode_image(blob)
    image = tf.image.resize_with_pad(image, 256, 256)
    if is_test:
        image = tf.image.crop_to_bounding_box(image, 16, 16, 224, 224)
    else:
        image = tf.image.random_crop(image, (224, 224, 3))
        image = tf.image.random_flip_left_right(image)
        image = tf.image.random_brightness(image, .05)
    return image


def Garbage(list_file, batch_size=32, is_test=False):
    shuffle = not is_test
    return_weights = not is_test

    root_dir = os.path.dirname(list_file)
    list_ = []
    with open(list_file) as f:
        for line in f.readlines():
            path, label = line.split(', ')
            path = f'{root_dir}/train_data/{path}'
            list_.append((str(path), int(label)))

    paths, labels = zip(*list_)
    path_ds = tf.data.Dataset.from_tensor_slices(list(paths))
    image_ds = path_ds.map(partial(preprocess_image, is_test=is_test), num_parallel_calls=AUTOTUNE)
    label_ds = tf.data.Dataset.from_tensor_slices(tf.cast(labels, tf.int64))  # .map(lambda y: label1[y], label)
    ds = tf.data.Dataset.zip((image_ds, label_ds))

    if shuffle:
        ds = ds.shuffle(buffer_size=len(list_))
    ds = ds.batch(batch_size).prefetch(buffer_size=AUTOTUNE)

    if return_weights:
        freq = np.bincount(labels)
        ds = ds, freq / freq.sum()

    return ds
