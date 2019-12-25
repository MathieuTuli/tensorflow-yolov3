from tensorflow.keras import layers, models
from typing import Union, Tuple

import tensorflow as tf
import logging

from .helpers import BatchNormalization


class Darknet53():
    def __init__(self,) -> None:
        self.model: models.Model = None

    def __str__(self) -> str:
        return 'Darknet53'

    @staticmethod
    def conv(prev_layer: layers.Layer,
             filters: int,
             kernel_size: Union[int, Tuple[int, int]],
             strides:  Union[int, Tuple[int, int]] = 1,
             data_format: str = "channels_last",
             batch_norm: bool = True) -> layers.Layer:
        if strides == 1:
            padding = 'same'
        else:
            # top left half padding
            prev_layer = layers.ZeroPadding2D(((1, 0), (1, 0)))(prev_layer)
            padding = 'valid'
        layer = layers.Conv2D(
            filters=filters,
            kernel_size=kernel_size,
            strides=strides,
            padding=padding,
            data_format=data_format,
            use_bias=not batch_norm,
            kernel_regularizer=tf.keras.regularizers.l2(0.0005))(prev_layer)
        if batch_norm:
            layer = BatchNormalization()(layer)
            layer = layers.LeakyReLU(alpha=0.1)(layer)

        return layer

    @staticmethod
    def block(prev_layer: layers.Layer,
              filters: int,
              num_blocks: int) -> layers.Layer:
        layer = Darknet53.conv(prev_layer=prev_layer,
                               filters=filters,
                               kernel_size=3,
                               strides=2)
        for i in range(num_blocks):
            layer = Darknet53.residual(layer, filters)
        return layer

    @staticmethod
    def residual(
            prev_layer: layers.Layer,
            filters: int) -> layers.Layer:
        res = Darknet53.conv(prev_layer=prev_layer,
                             filters=filters // 2,
                             kernel_size=1)
        res = Darknet53.conv(prev_layer=res,
                             filters=filters,
                             kernel_size=3)
        res = layers.Add()([prev_layer, res])
        return res

    def build(self) -> models.Model:
        # channel last
        layer = inputs = layers.Input([None, None, 3])
        layer = Darknet53.conv(prev_layer=layer,
                               filters=32,
                               kernel_size=3)
        layer = Darknet53.block(prev_layer=layer,
                                filters=64,
                                num_blocks=1)
        layer = layer_36 = Darknet53.block(prev_layer=layer,
                                           filters=256,
                                           num_blocks=8)
        layer = layer_61 = Darknet53.block(prev_layer=layer,
                                           filters=512,
                                           num_blocks=8)
        layer = Darknet53.block(layer, filters=1024, num_blocks=4)
        self.model = models.Model(inputs,
                                  (layer_36, layer_61, layer),
                                  name=str(self))
        return self.model

    def summary(self) -> str:
        if self.model is None:
            return 'Model is None'
        return self.model.summary()
