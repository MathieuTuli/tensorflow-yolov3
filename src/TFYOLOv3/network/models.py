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

    def __init__(self,
                 num_classes: int = -1,
                 backbone: bool = True,
                 **kwargs) -> None:
        '''
        @param: num_classes: if backbone is set, must also set
        @param: backbone: flag to indicate use of Darknet53 as a backbone
                          feature extractor
        '''
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
        layer = early_stop = self.block3(layer)
        layer = middle_stop = self.block4(layer)
        layer = self.block5(layer)
        if self.pooled is not None:
            layer = self.pooled(layer)
            layer = self.dense(layer)
        return (early_stop, middle_stop, layer)


class YOLOv3(Model):
    def __init__(self,
                 num_classes: int,
                 backbone: bool = True,
                 **kwargs) -> None:
        super(YOLOv3, self).__init__(**kwargs)
        self.conv_scale3 = Darknet53Conv(filters=255, kernel_size=1)

        # TODO: add upsample
        self.conv1_scale2 = Darknet53Conv(filters=256, kernel_size=1)
        self.conv2_scale2 = Darknet53Conv(filters=255, kernel_size=1)
        self.block1_scale2 = [layers.Add()[
            Darknet53Conv(filters=256, kernel_size=1),
            Darknet53Conv(filters=512, kernel_size=3)] for i in range(3)]

        # TODO: add upsample
        self.conv1_scale1 = Darknet53Conv(filters=128, kernel_size=1)
        self.conv2_scale1 = Darknet53Conv(filters=255, kernel_size=1)
        self.block1_scale1 = [layers.Add()[
            Darknet53Conv(filters=128, kernel_size=1),
            Darknet53Conv(filters=256, kernel_size=3)] for i in range(3)]
