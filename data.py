import glob

import tensorflow as tf

AUTOTUNE = tf.data.experimental.AUTOTUNE

label1 = sum([[i] * c for i, c, in enumerate([6, 8, 23, 3])], [])


def preprocess_image(path):
    blob = tf.io.read_file(path)
    image = tf.image.decode_image(blob)
    image = tf.image.resize_with_pad(image, 256, 256)
    image = tf.image.random_crop(image, (224, 224, 3))
    return image


def Garbage(root_dir, batch_size=32, shuffle=True):
    list_ = []
    for file in glob.glob(f'{root_dir}/train_data/*.txt'):
        with open(file) as f:
            path, label = f.readline().split(', ')
            path = f'{root_dir}/train_data/{path}'
        list_.append((str(path), int(label)))

    paths, labels = zip(*list_)
    path_ds = tf.data.Dataset.from_tensor_slices(list(paths))
    image_ds = path_ds.map(preprocess_image, num_parallel_calls=AUTOTUNE)
    label_ds = tf.data.Dataset.from_tensor_slices(tf.cast(labels, tf.int64))  # .map(lambda y: label1[y], label)
    ds = tf.data.Dataset.zip((image_ds, label_ds))

    if shuffle:
        ds = ds.shuffle(buffer_size=len(list_))
    ds = ds.batch(batch_size).prefetch(buffer_size=AUTOTUNE)

    return ds
