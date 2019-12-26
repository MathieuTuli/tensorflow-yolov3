from tensorflow.keras import layers, models, Model
from typing import Union, Tuple, List

import tensorflow as tf
import logging

from .helpers import BatchNormalization


class Darknet53Conv(layers.Layer):
    def __init__(self,
                 filters: int,
                 kernel_size: Union[int, Tuple[int, int]],
                 strides:  Union[int, Tuple[int, int]] = 1,
                 data_format: str = "channels_last",
                 batch_norm: bool = True,
                 kernel_regularizer: float = 0.0005, **kwargs):
        super(Darknet53Conv, self).__init__(**kwargs)
        padding = 'same' if strides == 1 else 'valid'
        self.layer = layers.Conv2D(
            filters=filters,
            kernel_size=kernel_size,
            strides=strides,
            padding=padding,
            data_format=data_format,
            use_bias=not batch_norm,
            kernel_regularizer=tf.keras.regularizers.l2(kernel_regularizer))
        self.batch_norm = None
        self.leaky_relu = None
        if batch_norm:
            self.batch_norm = BatchNormalization()
            self.leaky_relu = layers.LeakyReLU(alpha=0.1)

    def call(self,
             inputs: Union[tf.Tensor,
                           Tuple[tf.Tensor, ...],
                           List[tf.Tensor]],
             **kwargs) -> Union[tf.Tensor,
                                Tuple[tf.Tensor, ...],
                                List[tf.Tensor]]:
        layer = self.layer(inputs)
        if self.batch_norm is not None:
            layer = self.batch_norm(layer)
            layer = self.leaky_relu(layer)
        return layer


class Darknet53Residual(layers.Layer):
    def __init__(self, filters: int, **kwargs):
        super(Darknet53Residual, self).__init__(**kwargs)
        self.res1 = Darknet53Conv(filters=filters // 2,
                                  kernel_size=1)
        self.res2 = Darknet53Conv(filters=filters,
                                  kernel_size=3)

    def call(self,
             inputs: Union[tf.Tensor,
                           Tuple[tf.Tensor, ...],
                           List[tf.Tensor]],
             **kwargs) -> Union[tf.Tensor,
                                Tuple[tf.Tensor, ...],
                                List[tf.Tensor]]:
        layer = self.res1(inputs)
        layer = self.res2(layer)
        # shortcut
        return layers.Add()([inputs, layer])


class Darknet53Block(layers.Layer):
    def __init__(self,
                 filters: int,
                 num_res: int,
                 conv_strides: Union[int, Tuple[int, int]] = 2,
                 **kwargs):
        super(Darknet53Block, self).__init__(**kwargs)
        self.conv = Darknet53Conv(filters=filters,
                                  kernel_size=3,
                                  strides=2)
        self.res_blocks = [Darknet53Residual(filters=filters)
                           for x in range(num_res)]

    # def build(self,
    #           input_shape: Tuple[int, int, int],
    #           **kwargs) -> Union[tf.Tensor,
    #                              Tuple[tf.Tensor, ...],
    #                              List[tf.Tensor]]:
    #     ...

    def call(self,
             inputs: Union[tf.Tensor,
                           Tuple[tf.Tensor, ...],
                           List[tf.Tensor]],
             **kwargs) -> Union[tf.Tensor,
                                Tuple[tf.Tensor, ...],
                                List[tf.Tensor]]:
        layer = self.conv(inputs)
        for res in self.res_blocks:
            layer = res(layer)
        return layer


class Darknet53(Model):
    """
    as defined in:
        https://github.com/pjreddie/darknet/blob/master/cfg/darknet53.cfg
        https://pjreddie.com/media/files/papers/YOLOv3.pdf
    """

    def __init__(self, input_shape: Tuple[int, int, int],
                 num_classes: int, backbone: bool = True, **kwargs) -> None:
        """
        @param input_shape: requires channels_last specfication
        """
        super(Darknet53, self).__init__(**kwargs)

        logging.info("Darknet53: starting build")
        self.conv1 = Darknet53Conv(filters=32, kernel_size=3)
        self.block1 = Darknet53Block(filters=64, num_res=1)
        self.block2 = Darknet53Block(filters=128, num_res=2)
        self.block3 = Darknet53Block(filters=256, num_res=8)
        self.block4 = Darknet53Block(filters=512, num_res=8)
        self.block5 = Darknet53Block(filters=1024, num_res=4)
        self.pooled = None if backbone else layers.GlobalAveragePooling2D(
            data_format='channels_last')
        self.dense = None if backbone else layers.Dense(
            num_classes, activation='softmax')
        logging.info(
            "Darknet53: model built successfully. Call " +
            "*model*.summary() for summary")

    def call(self,
             inputs: Union[tf.Tensor,
                           Tuple[tf.Tensor, ...],
                           List[tf.Tensor]],
             training: bool = False) -> Union[tf.Tensor,
                                              Tuple[tf.Tensor, ...],
                                              List[tf.Tensor]]:
        layer = self.conv1(inputs)
        layer = self.block1(layer)
        layer = self.block2(layer)
        layer = skip1 = self.block3(layer)
        layer = skip2 = self.block4(layer)
        layer = self.block5(layer)
        if self.pooled:
            layer = self.pooled(layer)
            layer = self.dense(layer)
        return (skip1, skip2, layer)
