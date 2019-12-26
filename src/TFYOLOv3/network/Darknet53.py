from tensorflow.keras import layers, models
from typing import Union, Tuple

import tensorflow as tf
import logging

from .helpers import BatchNormalization


class Darknet53Conv(layers.Layer):
    def __init__(self):
        super(Darknet53Conv, self).__init__()


class Darknet53Residual(layers.Layer):
    def __init__(self):
        super(Darknet53Residual, self).__init__()


class Darknet53Block(layers.Layer):
    def __init__(self):
        super(Darknet53Block, self).__init__()


class Darknet53(models.Model):
    """
    as defined in:
        https://github.com/pjreddie/darknet/blob/master/cfg/darknet53.cfg
        https://pjreddie.com/media/files/papers/YOLOv3.pdf
    """

    def __init__(self, name: str = "Darknet53") -> None:
        super(Darknet53, self).__init__()
        self.inputs, self.skip1, self.skip2, self.layer = self.build()

    def __str__(self) -> str:
        return self.name

    @staticmethod
    def conv(prev_layer: layers.Layer,
             filters: int,
             kernel_size: Union[int, Tuple[int, int]],
             strides:  Union[int, Tuple[int, int]] = 1,
             data_format: str = "channels_last",
             batch_norm: bool = True,
             kernel_regularizer: float = 0.0005) -> layers.Layer:
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
            kernel_regularizer=tf.keras.regularizers.l2(
                kernel_regularizer))(prev_layer)
        if batch_norm:
            layer = BatchNormalization()(layer)
            layer = layers.LeakyReLU(alpha=0.1)(layer)

        return layer

    @staticmethod
    def residual(prev_layer: layers.Layer,
                 filters: int) -> layers.Layer:
        res = Darknet53.conv(prev_layer=prev_layer,
                             filters=filters // 2,
                             kernel_size=1)
        res = Darknet53.conv(prev_layer=res,
                             filters=filters,
                             kernel_size=3)
        return layers.Add()([prev_layer, res])

    @staticmethod
    def block(prev_layer: layers.Layer,
              filters: int,
              num_res: int,
              conv_strides: Union[int, Tuple[int, int]] = 2,) -> layers.Layer:
        layer = Darknet53.conv(prev_layer=prev_layer,
                               filters=filters,
                               kernel_size=3,
                               strides=2)
        for i in range(num_res):
            layer = Darknet53.residual(layer, filters)
        return layer

    @tf.function()
    def build(self) -> models.Model:
        # channel last
        logging.info("Darknet53: starting build")
        layer = inputs = layers.Input([None, None, 3])
        layer = Darknet53.conv(prev_layer=layer,
                               filters=32,
                               kernel_size=3)
        layer = Darknet53.block(prev_layer=layer,
                                filters=64,
                                num_res=1)
        layer = Darknet53.block(prev_layer=layer,
                                filters=128,
                                num_res=2)
        layer = skip1 = Darknet53.block(prev_layer=layer,
                                        filters=256,
                                        num_res=8)
        layer = skip2 = Darknet53.block(prev_layer=layer,
                                        filters=512,
                                        num_res=8)
        layer = Darknet53.block(layer,
                                filters=1024,
                                num_res=4)
        logging.info(
            "Darknet53: model built successfully. Call " +
            "*model*.summary() for summary")
        return inputs, skip1, skip2, layer

    def call(self, inputs):
