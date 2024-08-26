"""
Load the digits dataset
"""

import os

os.environ["KERAS_BACKEND"] = "jax"
import keras
import jax.numpy as jnp  # JAX NumPy
import numpy as np


class MNISTData(keras.utils.Sequence):
    def __init__(self, x_in, y_in, batch_size, shuffle=True):
        # Initialization
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.x = jnp.expand_dims(jnp.array(x_in), -1) / 255
        self.y = jnp.array(y_in)
        self.datalen = len(y_in)
        self.on_epoch_end()

    def __getitem__(self, index):
        # get batch indexes from shuffled indexes
        batch_indexes = self.indexes[
            index * self.batch_size : (index + 1) * self.batch_size
        ]
        x_batch = self.x[batch_indexes]
        y_batch = self.y[batch_indexes]
        return x_batch, y_batch

    def __len__(self):
        # Denotes the number of batches per epoch
        return self.datalen // self.batch_size

    def __iter__(self):
        for i in range(len(self)):
            yield self[i]

    def on_epoch_end(self):
        # Updates indexes after each epoch
        self.indexes = np.arange(self.datalen)
        if self.shuffle:
            np.random.shuffle(self.indexes)


def get_raw_digits_dataset():
    (x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data(
        path="mnist.npz"
    )
    return (x_train, y_train), (x_test, y_test)


def get_train_test_keras_datasets(batch_size=32):
    (x_train, y_train), (x_test, y_test) = get_raw_digits_dataset()
    train = MNISTData(x_train, y_train, batch_size)
    test = MNISTData(x_test, y_test, batch_size)
    return train, test
