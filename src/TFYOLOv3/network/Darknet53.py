from tensorflow.keras import layers, models

import tensorflow as tf
import logging


class Darknet53():
    def __init__(self,) -> None:
        ...

    def __str__(self) -> str:
        return 'Darknet53'

    @staticmethod
    def conv() -> layers.Layer:
        ...

    @staticmethod
    def block() -> layers.Layer:
        ...

    @staticmethod
    def residual() -> layers.Layer:
        ...

    def build(self, eager: bool = False) -> models.Model:
        model = models.Sequential()
        if eager:
            ...
        else:
            model.add(layers.Conv2D(filters=32, kernel_size=(3, 3),
                                    strides=(1, 1), padding='valid',
                                    data_format='channels_last',
                                    activation='',
                                    input_shape=(256, 256, 3)))
            model.add(layers.Conv2D(filters=64, kernel_size=(3, 3),
                                    strides=(2, 2), padding='valid',
                                    data_format='channels_last',
                                    activation=''))
