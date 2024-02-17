from tensorflow import keras
from tensorflow.keras import  layers, regularizers
from keras.layers import *
import tensorflow as tf

class SPADE(layers.Layer):
    """
    Implementation of Spatially-Adaptive Normalization layer
    
    """
    def __init__(self, filters, epsilon=1e-5, **kwargs):
        super().__init__(**kwargs)
        self.epsilon = epsilon
        self.conv = layers.Conv2D(filters, 3, padding="same", activation="relu")
        self.conv_gamma = layers.Conv2D(filters, 3, padding="same")
        self.conv_beta = layers.Conv2D(filters, 3, padding="same")

    def build(self, input_shape):
        self.resize_shape = input_shape[1:3]

    def call(self, input_tensor, raw_mask):
        with tf.device('/GPU:0'):

            mask = tf.image.resize(raw_mask, self.resize_shape, method="nearest")
            x = self.conv(mask)    

            gamma = self.conv_gamma(x)
            beta = self.conv_beta(x)
            mean, var = tf.nn.moments(input_tensor, axes=(0, 1, 2), keepdims=True)
            std = tf.sqrt(var + self.epsilon)

            normalized = (input_tensor - mean) / std
            output = gamma * normalized + beta
            
        return output

def spade_generator_unit(x, mask, feats_out, kernel, upsampling = True):
    """
    SPADE block: x_in -> GaussianNoise ---> (SPADE + Relu + Conv) x 2 -----> upsampling (optional) --> output
                                        |                               |
                                         ---- (SPADE + Relu + Conv) ----
    """
    x = GaussianNoise(0.05)(x)
    
    # Residual SPADE & conv
    spade1 = SPADE(feats_out)(x, mask)
    relu1 = LeakyReLU(0.2)(spade1)
    conv1 = Conv2D(feats_out,kernel, padding='same')(relu1)
    spade2 = SPADE(feats_out)(conv1, mask)
    relu2 = LeakyReLU(0.2)(spade2)
    conv2 = Conv2D(feats_out,kernel, padding='same')(relu2)
    
    # Skip
    spade_skip = SPADE(feats_out)(x, mask)
    relu_skip = LeakyReLU(0.2)(spade_skip)
    conv_skip = Conv2D(feats_out,kernel, padding='same')(relu_skip)

    # Add 
    output = tf.keras.layers.add([conv_skip,conv2])
    if upsampling == True:
        output = UpSampling2D(size = (2,2),interpolation='bilinear')(output)
        return output
    else:
        return output