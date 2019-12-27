from tensorflow.keras import layers, Model
from typing import Union, Tuple, List

import tensorflow as tf
import logging

from .layers import (
    Darknet53Conv,
    Darknet53Block,
    YOLOConvUpsample,
    YOLOBlock,
    YOLOOutput)


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
        if self.pooled is not None and self.dense is not None:
            layer = self.pooled(layer)
            layer = self.dense(layer)
        return (early_stop, middle_stop, layer)


class YOLOv3(Model):
    def __init__(
            self,
            num_classes: int,
            anchors: List[Tuple[int, int]] = [
                (10, 13), (16, 30), (33, 23), (30, 61), (62, 45),
                (59, 119), (116, 90), (156, 198), (373, 326)],
            masks: List[Tuple[int, int, int]] = [
                (0, 1, 2), (3, 4, 5), (6, 7, 8)],
            **kwargs) -> None:
        super(YOLOv3, self).__init__(**kwargs)
        self.backbone = Darknet53()

        self.scale1_upsampler = YOLOConvUpsample(filters=128)
        self.scale1_block = YOLOBlock(filters=128)
        self.scale1_conv = Darknet53Conv(filters=255,
                                         kernel_size=1,
                                         batch_norm=False,
                                         activation="Linear")
        self.scale1_ouptut = YOLOOutput(filters=128,
                                        num_anchors=len(masks[0]),
                                        num_classes=num_classes)

        self.scale2_upsampler = YOLOConvUpsample(filters=256)
        self.scale2_block = YOLOBlock(filters=256)
        self.scale2_conv = Darknet53Conv(filters=255,
                                         kernel_size=1,
                                         batch_norm=False,
                                         activation="Linear")
        self.scale2_ouptut = YOLOOutput(filters=256,
                                        num_anchors=len(masks[1]),
                                        num_classes=num_classes)

        self.scale3_block = YOLOBlock(filters=512)
        self.scale3_conv = Darknet53Conv(filters=255,
                                         kernel_size=1,
                                         batch_norm=False,
                                         activation="Linear")
        self.scale3_ouptut = YOLOOutput(filters=512,
                                        anchors=len(masks[2]),
                                        num_classes=num_classes)
        self.anchors = anchors
        self.masks = masks
        self.num_classes = num_classes

    def call(self,
             inputs: Union[tf.Tensor,
                           Tuple[tf.Tensor, ...],
                           List[tf.Tensor]],
             training: bool = False) -> Union[tf.Tensor,
                                              Tuple[tf.Tensor, ...],
                                              List[tf.Tensor]]:
        scale1_skip, scale2_skip, layer = self.darknet_backbone.call(
            inputs, training)
        scale3_layer = scale2_layer = self.scale3_block(layer)
        scale3_layer = self.scale3_conv(scale3_layer)
        scale3_output = self.scale3_output(scale3_layer)
        scale2_layer = self.scale2_upsampler(scale2_layer)
        scale2_layer = layers.Concatenate(axis=-1)([scale2_layer,
                                                    scale2_skip])
        scale2_layer = self.scale2_block(scale2_layer)
        scale2_layer = self.scale2_conv(scale2_layer)
        scale2_output = self.scale2_output(scale2_layer)
        scale1_layer = self.scale1_upsampler(scale2_layer)
        scale1_layer = layers.Concatenate(axis=-1)([scale1_layer,
                                                    scale1_skip])
        scale1_layer = self.scale1_block(scale1_layer)
        scale1_layer = self.scale1_conv(scale1_layer)
        scale1_output = self.scale1_output(scale1_layer)
        if training:
            return scale1_output, scale2_output, scale3_output
        outputs = YOLOv3.infer(scale1_layer=scale1_output,
                               scale2_layer=scale2_output,
                               scale3_layer=scale3_output,
                               anchors=self.anchors,
                               masks=self.masks,
                               num_classes=self.num_classes)
        return outputs

    @staticmethod
    @tf.function
    def infer(scale1_layer: tf.Tensor,
              scale2_layer: tf.Tensor,
              scale3_layer: tf.Tensor,
              anchors: List[Tuple[int, int]],
              masks: List[Tuple[int, int, int]],
              num_classes: int) -> Union[tf.Tensor,
                                         Tuple[tf.Tensor, ...],
                                         List[tf.Tensor]]:
        ...
