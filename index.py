import os

import tensorflow_hub as hub
from tensorflow.python.keras import Sequential
from tensorflow.python.keras.activations import softmax
from tensorflow.python.keras.layers import Dense
from tensorflow.python.keras.losses import sparse_categorical_crossentropy
from tensorflow.python.keras.metrics import sparse_categorical_accuracy
from tensorflow.python.keras.optimizer_v2.adam import Adam

from data import Garbage

model = Sequential([
    hub.KerasLayer("https://tfhub.dev/google/imagenet/mobilenet_v2_100_224/classification/4", name='mobilenetv2'),
    Dense(40, activation=softmax, name='fc'),
], name='classifier')
model.build([None, 224, 224, 3])  # Batch input shape.
model.summary()

generator = Garbage(os.path.expanduser('~/garbage_classify'))
model.compile(optimizer=Adam(), loss=sparse_categorical_crossentropy, metrics=[sparse_categorical_accuracy],)
model.fit_generator(generator, epochs=20)
