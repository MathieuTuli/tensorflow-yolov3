import tensorflow as tf


class BatchNormalization(tf.keras.layers.BatchNormalization):
    def call(self,
             layer: tf.keras.layers.Layer,
             training: bool = False):
        if training is None:
            training = tf.constant(False)
        return super().call(layer, training)
