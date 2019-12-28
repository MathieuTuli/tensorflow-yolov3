from tensorflow.keras import layers
from typing import Union, Tuple, List

import tensorflow as tf
import logging


class BatchNormalization(tf.keras.layers.BatchNormalization):
    """@credit: https://github.com/zzh8829
                https://github.com/zzh8829/yolov3-tf2/blob/master/yolov3_tf2/batch_norm.py
    """

    def call(self,
             inputs: Union[tf.Tensor, Tuple[tf.Tensor, ...],
                           List[tf.Tensor]],
             training: bool = False) -> Union[tf.Tensor,
                                              Tuple[tf.Tensor, ...],
                                              List[tf.Tensor]]:
        if training is None:
            training = tf.constant(False)
        return super().call(inputs, training)


class Darknet53Conv(layers.Layer):
    def __init__(self,
                 filters: int,
                 kernel_size: Union[int, Tuple[int, int]],
                 strides:  Union[int, Tuple[int, int]] = 1,
                 data_format: str = "channels_last",
                 batch_norm: bool = True,
                 kernel_regularizer: float = 0.0005,
                 activation: str = 'LeakyReLU', **kwargs) -> None:
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
        self.activation = None
        # TODO use activation in Conv2D and avoid below
        if batch_norm:
            self.batch_norm = BatchNormalization()
            if activation == 'LeakyReLU':
                self.activation = layers.LeakyReLU(alpha=0.1)
            elif activation == 'Linear':
                ...
            else:
                logging.critical("Darknet53Conv: Unknown activation")
                raise

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
        if self.activation is not None:
            layer = self.activation(layer)
        return layer


class Darknet53Residual(layers.Layer):
    def __init__(self, filters: int, **kwargs) -> None:
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
                 **kwargs) -> None:
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


class YOLOConvUpsample(layers.Layer):
    def __init__(self,
                 filters: int,
                 **kwargs) -> None:
        super(YOLOConvUpsample, self).__init__(**kwargs)
        self.conv = Darknet53Conv(filters=filters, kernel_size=1)
        self.upsample = layers.UpSampling2D(
            size=2, data_format='channels_last',
            interpolation='bilinear')

    def call(self,
             inputs: Union[tf.Tensor,
                           Tuple[tf.Tensor, ...],
                           List[tf.Tensor]],
             **kwargs) -> Union[tf.Tensor,
                                Tuple[tf.Tensor, ...],
                                List[tf.Tensor]]:
        layer = self.conv(inputs)
        layer = self.upsample(layer)
        return layer


class YOLOBlock(layers.Layer):
    def __init__(self,
                 filters: int,
                 **kwargs) -> None:
        super(YOLOBlock, self).__init__(**kwargs)
        self.conv1 = Darknet53Conv(filters=filters, kernel_size=1)
        self.convs = [
            Darknet53Conv(filters=filters * 2, kernel_size=3),
            Darknet53Conv(filters=filters, kernel_size=1),
            Darknet53Conv(filters=filters * 2, kernel_size=3),
            Darknet53Conv(filters=filters, kernel_size=1),
            Darknet53Conv(filters=filters * 2, kernel_size=3)]

    def call(self,
             inputs: Union[tf.Tensor,
                           Tuple[tf.Tensor, ...],
                           List[tf.Tensor]],
             **kwargs) -> Union[tf.Tensor,
                                Tuple[tf.Tensor, ...],
                                List[tf.Tensor]]:
        layer = self.conv1(inputs)
        for conv in self.convs:
            layer = conv(layer)
        return layer


class YOLOOutput(layers.Layer):
    def __init__(self,
                 filters: int,
                 num_classes: int,
                 num_anchors: int,
                 **kwargs) -> None:
        super(YOLOOutput, self).__init__(**kwargs)
        self.conv1 = Darknet53Conv(filters=filters * 2, kernel_size=3)
        self.conv2 = Darknet53Conv(filters=num_anchors * (num_classes + 5),
                                   kernel_size=1,
                                   batch_norm=False,
                                   activation='Linear')
        self.num_anchors = num_anchors
        self.num_classes = num_classes

    def call(self,
             inputs: Union[tf.Tensor,
                           Tuple[tf.Tensor, ...],
                           List[tf.Tensor]],
             **kwargs) -> Union[tf.Tensor,
                                Tuple[tf.Tensor, ...],
                                List[tf.Tensor]]:
        """
        @credit: https://github.com/zzh8829/yolov3-tf2/blob/master/yolov3_tf2/models.py
        """
        layer = self.conv1(inputs)
        layer = self.conv2(layer)
        layer = layers.Lambda(lambda layer: tf.reshape(
            layer,
            (-1,
             tf.shape(layer)[1],
             tf.shape(layer)[2],
             self.num_anchors,
             self.num_classes + 5)))(layer)
        return layer


class YOLOInference(layers.Layer):
    def __init__(self,
                 num_classes: int,
                 anchors: List[Tuple[int, int]],
                 masks: List[Tuple[int, int, int]],
                 **kwargs) -> None:
        super(YOLOInference, self).__init__(**kwargs)
        self.num_classes = num_classes
        self.anchors = anchors
        self.masks = masks

    def call(self,
             inputs: Union[tf.Tensor,
                           Tuple[tf.Tensor, ...],
                           List[tf.Tensor]],
             **kwargs) -> Union[tf.Tensor,
                                Tuple[tf.Tensor, ...],
                                List[tf.Tensor]]:
        """
        """
        return layer
