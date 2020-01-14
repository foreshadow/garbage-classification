import os
from datetime import datetime

import tensorflow_hub as hub
from tensorflow.python.keras import Sequential
from tensorflow.python.keras.activations import relu, softmax
from tensorflow.python.keras.callbacks import ModelCheckpoint, ReduceLROnPlateau
from tensorflow.python.keras.layers import Dense
from tensorflow.python.keras.losses import sparse_categorical_crossentropy
from tensorflow.python.keras.metrics import sparse_categorical_accuracy
from tensorflow.keras.optimizers import Adam

from data import Garbage

model = Sequential([
    hub.KerasLayer("https://tfhub.dev/google/imagenet/mobilenet_v2_100_224/classification/4", name='mobile_net_v2'),
    Dense(256, activation=relu),
    Dense(40, activation=softmax),
])
model.build((None, 224, 224, 3))
model.summary()

train = Garbage(os.path.expanduser('~/garbage_classify/train.lst'), shuffle=True)
val = Garbage(os.path.expanduser('~/garbage_classify/val.lst'))
model.compile(
    optimizer=Adam(),
    loss=sparse_categorical_crossentropy,
    metrics=[sparse_categorical_accuracy],
)
model.fit(train, epochs=300, class_weight=None, validation_data=val, callbacks=[
    ModelCheckpoint(os.path.join(
        'checkpoints',
        datetime.now().strftime('%m%d%H%M%S'),
        'mobilenetv2-40-ep{epoch:02d}-train{sparse_categorical_accuracy:.4f}-val{val_sparse_categorical_accuracy:.4f}.ckpt'
    )),
    ReduceLROnPlateau(monitor='loss', patience=5, factor=.2, verbose=True),
])
