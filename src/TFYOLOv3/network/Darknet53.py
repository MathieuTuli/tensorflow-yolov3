from tensorflow.keras import layers, models, Sequential

import tensorflow as tf
import logging


class Darknet53():
    def __init__(self,) -> None:
        ...

    def __str__(self) -> str:
        return 'Darknet53'

    def build(self, eager: bool = False) -> Sequential:
        model = models.Sequential()
        if eager:
            ...
        else:
            model.add(layers.Conv2D(filters=32, kernel_size=(3, 3),
                                    strides=(1, 1), padding='valid',
                                    data_format='channels_last',
                                    activation='', input_shape=(, ,)))
            model.add(layers.())
            model.add(layers.())
