import glob
import random

import tensorflow as tf
from skimage.io import imread
from tensorflow.python.keras.utils import Sequence


class Garbage(Sequence):
    def __init__(self, root_dir, batch_size=32, shuffle=True):
        self.root_dir = root_dir
        self.batch_size = batch_size
        self.shuffle = shuffle
        list_ = []
        for file in glob.glob(f'{root_dir}/train_data/*.txt'):
            with open(file) as f:
                name, label = f.readline().split(', ')
            list_.append((str(name), int(label)))
        self.list = list_

    def __len__(self):
        return len(self.list) // self.batch_size

    def __getitem__(self, item):
        images = []
        labels = []
        for name, label in self.list[item * self.batch_size: (item + 1) * self.batch_size]:
            image = imread(f'{self.root_dir}/train_data/{name}')
            h, w, c = image.shape
            target = max(h, w)
            image = tf.image.pad_to_bounding_box(image, (target - h) // 2, (target - w) // 2, target, target)
            image = tf.image.resize(image, (256, 256))
            image = tf.image.random_crop(image, (224, 224, 3))
            images.append(image)
            labels.append(label)
        return images, tf.constant(labels)

    def on_epoch_end(self):
        if self.shuffle:
            random.shuffle(self.list)
