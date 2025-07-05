import tensorflow as tf
from tensorflow.keras.layers import Input, Conv2D, ReLU, Lambda
from tensorflow.keras.models import Model

def SubPixelConv2D(scale):
    return Lambda(lambda x: tf.nn.depth_to_space(x, scale))

def build_espcn(scale=4, input_shape=(200, 300, 3)):
    inputs = Input(shape=input_shape)
    
    x = Conv2D(64, (5, 5), padding='same')(inputs)
    x = ReLU()(x)

    x = Conv2D(32, (3, 3), padding='same')(x)
    x = ReLU()(x)

    x = Conv2D(3 * (scale ** 2), (3, 3), padding='same')(x)
    outputs = SubPixelConv2D(scale)(x)
    
    model = Model(inputs=inputs, outputs=outputs)
    return model
