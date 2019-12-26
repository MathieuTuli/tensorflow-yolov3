from tensorflow.keras import layers, Model
from typing import Union, Tuple, List

import tensorflow as tf
import logging

from .layers import Darknet53Conv, Darknet53Block


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
