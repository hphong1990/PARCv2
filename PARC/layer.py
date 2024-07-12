from tensorflow import keras
from tensorflow.keras import layers, regularizers
from keras.layers import *
import tensorflow as tf
import numpy as np

def resnet_unit(feat_dim, kernel_size, x_in, padding="CONSTANT"):
    """
    Resnet unit: x_in --> conv --> relu --> conv --> + relu --> x_out
                    |                           |
                     ---------------------------
    Parameter: 
                - feat_dim (int): number of channels
                - kernel_size (int): size of convolution kernel
                - x_in (tensor): input tensor
                - padding (str): padding method to use
    Return:
                - (tensor): output tensor of residual unit.
    """
    pad_size = (kernel_size - 1) // 2
    pad_instruct = tf.constant([[0, 0], [pad_size, pad_size], [pad_size, pad_size], [0, 0]])
    x = tf.pad(x_in, pad_instruct, padding)
    x = Conv2D(feat_dim, kernel_size, padding="valid")(x)
    x = ReLU()(x)
    x = tf.pad(x, pad_instruct, padding)
    x = Conv2D(feat_dim, kernel_size, padding="valid")(x)
    x = ReLU()(x_in + x)
    return x

def resnet_block(x_in, feat_dim, kernel_size, reps, pooling=True, padding="CONSTANT"):
    """
    Assembly of multiple resnet unit: x_in --> 2 x (conv + relu) --> 'reps' x resnet_unit --> output
    Parameter: 
                - x_in (tensor): input tensor
                - feat_dim (int): number of channels
                - kernel_size (int): size of convolution kernel
                - reps (int): number of residual unit
                - pooling (bool): the block ends with pooling or not
                - padding (str): padding method to use
    Return:
                - (tensor): output of the resnet block
    """
    pad_size = (kernel_size - 1) // 2
    pad_instruct = tf.constant([[0, 0], [pad_size, pad_size], [pad_size, pad_size], [0, 0]])
    x = tf.pad(x_in, pad_instruct, padding)
    x = Conv2D(feat_dim, kernel_size, padding="valid")(x)
    x = ReLU()(x)
    x = tf.pad(x, pad_instruct, padding)
    x = Conv2D(feat_dim, kernel_size, padding="valid")(x)
    x = ReLU()(x)
    for _ in range(reps):
        x = resnet_unit(feat_dim, kernel_size, x, padding)
    if pooling == True:
        x = MaxPooling2D(2,2)(x)
        return x
    else:
        return x

def conv_unit(feat_dim, kernel_size, x_in, padding="CONSTANT"):
    """
    Conv unit: x_in --> Conv k x k + relu --> Conv 1 x 1 + relu --> output
    Parameter: 
                - x_in (tensor): input tensor
                - feat_dim (int): number of channels
                - kernel_size (k) (int): size of convolution kernel
                - padding (str): padding method to use
    Return:
                - (tensor): output of the conv unit
    """
    pad_size = (kernel_size - 1) // 2
    pad_instruct = tf.constant([[0, 0], [pad_size, pad_size], [pad_size, pad_size], [0, 0]])
    x = tf.pad(x_in, pad_instruct, padding)
    x = Conv2D(feat_dim, kernel_size, activation=LeakyReLU(0.2), padding="valid")(x)
    x = Conv2D(feat_dim, 1, activation=LeakyReLU(0.2), padding="valid")(x)
    return x

def conv_block_down(x, feat_dim, reps, kernel_size, mode='normal', padding="CONSTANT"):
    if mode == 'down':
        x = MaxPooling2D(2,2)(x)
    for _ in range(reps):
        x = conv_unit(feat_dim, kernel_size, x, padding)
    return x

def conv_block_up_w_concat(x, x1, feat_dim, reps, kernel_size, mode='normal', padding="CONSTANT"):
    if mode == 'up':
        x = UpSampling2D((2,2),interpolation='bilinear')(x)
    x = Concatenate()([x,x1])
    for _ in range(reps):
        x = conv_unit(feat_dim, kernel_size, x, padding)
    return x

def conv_block_up_wo_concat(x, feat_dim, reps, kernel_size, mode='normal', padding="CONSTANT"):
    if mode == 'up':
        x = UpSampling2D((2,2),interpolation='bilinear')(x)
    for _ in range(reps):
        x = conv_unit(feat_dim, kernel_size, x, padding)
    return x


class SPADE(layers.Layer):
    """
    Implementation of Spatially-Adaptive Normalization layer
    
    """
    def __init__(self, filters, epsilon=1e-5, padding="CONSTANT", **kwargs):
        super().__init__(**kwargs)
        self.epsilon = epsilon
        self.conv = layers.Conv2D(filters, 3, padding="valid", activation="relu")
        self.conv_gamma = layers.Conv2D(filters, 3, padding="valid")
        self.conv_beta = layers.Conv2D(filters, 3, padding="valid")
        self.padding = padding

    def build(self, input_shape):
        self.resize_shape = input_shape[1:3]

    def call(self, input_tensor, raw_mask):
        with tf.device('/GPU:0'):

            mask = tf.image.resize(raw_mask, self.resize_shape, method="nearest")
            mask = tf.pad(mask, tf.constant([[0, 0], [1, 1], [1, 1], [0, 0]]), self.padding)
            x = self.conv(mask)    

            x = tf.pad(x, tf.constant([[0, 0], [1, 1], [1, 1], [0, 0]]), self.padding)
            gamma = self.conv_gamma(x)
            beta = self.conv_beta(x)
            mean, var = tf.nn.moments(input_tensor, axes=(0, 1, 2), keepdims=True)
            std = tf.sqrt(var + self.epsilon)

            normalized = (input_tensor - mean) / std
            output = gamma * normalized + beta
            
        return output

def spade_generator_unit(x, mask, feats_out, kernel, upsampling=True, padding="CONSTANT"):
    """
    SPADE block: x_in -> GaussianNoise ---> (SPADE + Relu + Conv) x 2 -----> upsampling (optional) --> output
                                        |                               |
                                         ---- (SPADE + Relu + Conv) ----
    """
    pad_size = (kernel - 1) // 2
    pad_instruct = tf.constant([[0, 0], [pad_size, pad_size], [pad_size, pad_size], [0, 0]])
    x = GaussianNoise(0.05)(x)
    # Residual SPADE & conv
    spade1 = SPADE(feats_out)(x, mask)
    relu1 = LeakyReLU(0.2)(spade1)
    relu1 = tf.pad(relu1, pad_instruct, padding)
    conv1 = Conv2D(feats_out, kernel, padding='valid')(relu1)
    spade2 = SPADE(feats_out, padding=padding)(conv1, mask)
    relu2 = LeakyReLU(0.2)(spade2)
    relu2 = tf.pad(relu2, pad_instruct, padding)
    conv2 = Conv2D(feats_out, kernel, padding='valid')(relu2)

    # Skip
    spade_skip = SPADE(feats_out, padding=padding)(x, mask)
    relu_skip = LeakyReLU(0.2)(spade_skip)
    relu_skip = tf.pad(relu_skip, pad_instruct, padding)
    conv_skip = Conv2D(feats_out,kernel, padding='valid')(relu_skip)

    # Add 
    output = tf.keras.layers.add([conv_skip,conv2])
    if upsampling == True:
        output = UpSampling2D(size = (2,2),interpolation='bilinear')(output)
        return output
    else:
        return output

    
def feature_extraction_resnet(input_shape=(128, 192), n_out_features=128, n_base_features=[32, 64], 
                              kernel_size=3, n_channel=5, padding="CONSTANT"):
    inputs = keras.Input(shape=(input_shape[0], input_shape[1], n_channel))
    conv = resnet_unit(n_base_features[0], kernel_size, inputs, padding=padding)
    for _ in range(1, len(n_base_features)):
        conv = resnet_unit(n_base_features[i], kernel_size, conv, padding=padding)
    feature_out = Conv2D(n_out_features, 1, padding="valid")(conv)
    unet = keras.Model(inputs, feature_out)
    return unet

    
def feature_extraction_unet(input_shape=(128,192), n_out_features=128, n_base_features=64, n_channel=5,
                            padding="CONSTANT"):
    inputs = keras.Input(shape = (input_shape[0], input_shape[1],n_channel))

    conv1 = conv_block_down(inputs,
                            feat_dim = n_base_features,
                            reps = 1,
                            kernel_size = 3,
                            padding = padding)
    conv2 = conv_block_down(conv1,
                            feat_dim = n_base_features*2,
                            reps = 1,
                            kernel_size = 3,
                            mode = 'down',
                            padding = padding)
    conv3 = conv_block_down(conv2,
                            feat_dim = n_base_features*4,
                            reps = 1,
                            kernel_size = 3,
                            mode = 'down',
                            padding = padding)
    conv4 = conv_block_down(conv3,
                            feat_dim = n_base_features*8,
                            reps = 1,
                            kernel_size = 3,
                            mode = 'down',
                            padding = padding)
    conv5 = conv_block_down(conv4,
                            feat_dim = n_base_features*16,
                            reps = 1,
                            kernel_size = 3,
                            mode = 'down',
                            padding = padding)
    conv6 = conv_block_up_wo_concat(conv5,
                                    feat_dim = n_base_features*8,
                                    reps = 1,
                                    kernel_size = 3,
                                    mode = 'up',
                                    padding = padding)
    conv7 = conv_block_up_w_concat(conv6, conv3,
                                    feat_dim = n_base_features*4,
                                    reps = 1,
                                    kernel_size = 3,
                                    mode = 'up',
                                    padding = padding)
    conv8 = conv_block_up_wo_concat(conv7,
                                    feat_dim = n_base_features*2,
                                    reps = 1,
                                    kernel_size = 3,
                                    mode = 'up',
                                    padding = padding)
    
    conv9 = conv_block_up_w_concat(conv8, conv1,
                                    feat_dim = n_out_features,
                                    reps = 1,
                                    kernel_size = 3,
                                    mode = 'up',
                                    padding = padding)
    feature_out = conv_block_up_wo_concat(conv9,
                                    feat_dim = n_out_features,
                                    reps = 1,
                                    kernel_size = 1,
                                    mode = 'normal',
                                    padding = padding)
    unet = keras.Model(inputs, feature_out)
    return unet

def feature_extraction_burgers(input_shape = (128,192), n_out_features = 64, n_base_features = 64, n_channel = 5):
    inputs = keras.Input(shape = (input_shape[0], input_shape[1],n_channel))
    # Need to put normalization layer
    conv1 = conv_block_down(inputs,
                            feat_dim = n_base_features,
                            reps = 1,
                            kernel_size = 3)
    conv2 = conv_block_down(conv1,
                            feat_dim = n_base_features*2,
                            reps = 1,
                            kernel_size = 3,
                            mode = 'down')
    conv3 = conv_block_down(conv2,
                            feat_dim = n_base_features*4,
                            reps = 1,
                            kernel_size = 3,
                            mode = 'down')
    conv8 = conv_block_up_wo_concat(conv3,
                                    feat_dim = n_base_features*2,
                                    reps = 1,
                                    kernel_size = 3,
                                    mode = 'up')
    conv9 = conv_block_up_w_concat(conv8, conv1,
                                    feat_dim = n_out_features,
                                    reps = 1,
                                    kernel_size = 3,
                                    mode = 'up')

    feature_out = conv_block_up_wo_concat(conv9,
                                    feat_dim = n_out_features,
                                    reps = 1,
                                    kernel_size = 1,
                                    mode = 'normal')
    unet = keras.Model(inputs, feature_out)
    return unet


class CentralDifference(object):
    def __init__(self, channel_size, filter_1d, padding):
        filter_size = filter_1d.shape[0]
        pad_size = (filter_size - 1) // 2
        if filter_size % 2:
            self.pad_instruct = tf.constant([[0, 0], [pad_size, pad_size], 
                                             [pad_size, pad_size], [0, 0]])
        else:
            self.pad_instruct = tf.constant([[0, 0], [pad_size, pad_size+1], 
                                             [pad_size, pad_size+1], [0, 0]])
        self.padding = padding
        # Construct dy filter
        self.dy_filter_qk = np.zeros((filter_size, filter_size, 
                                      channel_size, channel_size), dtype=np.float32)
        for i in range(channel_size):
            self.dy_filter_qk[:, pad_size, i, i] = filter_1d
        self.dy_filter_qk = tf.constant(self.dy_filter_qk)
        # Construct dx filter        
        self.dx_filter_qk = np.zeros((filter_size, filter_size, 
                                      channel_size, channel_size), dtype=np.float32)
        for i in range(channel_size):
            self.dx_filter_qk[pad_size, :, i, i] = filter_1d
        self.dx_filter_qk = tf.constant(self.dx_filter_qk)


    def __call__(self, array_2d):
        # Central differencing
        array_2d_padded = tf.pad(array_2d, self.pad_instruct, self.padding)
        dy = tf.nn.conv2d(array_2d_padded, self.dy_filter_qk, [1, 1, 1, 1], "VALID")
        dx = tf.nn.conv2d(array_2d_padded, self.dx_filter_qk, [1, 1, 1, 1], "VALID")
        return dy, dx


class Advection(layers.Layer):
    def __init__(self, channel_size=1, cd_filter_1d=np.array([-1.0, 1.0]), 
                 padding="SYMMETRIC", **kwargs):
        super().__init__(**kwargs)
        self.cdiff = CentralDifference(channel_size, cd_filter_1d, padding)

    def call(self, state_variable, velocity_field):
        # dy, dx = tf.image.image_gradients(state_variable)
        dy, dx = self.cdiff(state_variable)
        spatial_deriv = tf.concat([dy,dx],axis = -1)
        advect = tf.reduce_sum(tf.multiply(spatial_deriv, velocity_field),
                               axis=-1, keepdims=True)
        return advect

class Diffusion(layers.Layer):
    def __init__(self, channel_size=1, cd_filter_1d=tf.constant([-1.0, 1.0]), padding="SYMMETRIC", **kwargs):
        super().__init__(**kwargs)
        self.cdiff = CentralDifference(channel_size, cd_filter_1d, padding)

    def call(self, state_variable):
        # dy, dx = tf.image.image_gradients(state_variable)
        # dyy, _ = tf.image.image_gradients(dy)
        # _ , dxx = tf.image.image_gradients(dx)
        dy, dx = self.cdiff(state_variable)
        dyy, _ = self.cdiff(dy)
        _, dxx = self.cdiff(dx)
        laplacian = tf.add(dyy,dxx)
        return laplacian
    
class Poisson(layers.Layer):
    def __init__(self, channel_size=1, cd_filter_1d=tf.constant([-1.0, 1.0]), padding="SYMMETRIC", **kwargs):
        super().__init__(**kwargs)
        self.cdiff = CentralDifference(channel_size, cd_filter_1d, padding)

    def call(self, vector_field):
        # uy, ux = tf.image.image_gradients(vector_field[0])
        # vy, vx = tf.image.image_gradients(vector_field[1])
        uy, ux = self.cdiff(vector_field[0])
        vy, vx = self.cdiff(vector_field[1])
        ux2 = tf.multiply(ux, ux)
        vy2 = tf.multiply(vy, vy)
        uyvx = tf.multiply(uy, vx)
        return ux2, vy2, uyvx

def mapping_and_recon_cnn(input_shape=(128,192), n_base_features=128, n_mask_channel=1, 
                          output_channel=1, padding="CONSTANT"):
    inputs = keras.Input(shape = (input_shape[0], input_shape[1], n_base_features), dtype = tf.float32)
    inputs_2 = keras.Input(shape = (input_shape[0], input_shape[1], n_mask_channel), dtype = tf.float32)
    
    # Style vector 
    spade_1 = spade_generator_unit(inputs,
                                   inputs_2,
                                   n_base_features,
                                   1,
                                   upsampling = False,
                                   padding = padding)
    conv_last = resnet_block(spade_1, n_base_features, kernel_size = 1, reps = 2, pooling = False, padding = padding)

    conv_out = Conv2D(output_channel, 1, padding='valid')(conv_last)
    mapping_resnet = keras.Model([inputs,inputs_2], conv_out)
    return mapping_resnet

def integrator_cnn(input_shape = (128,192), n_base_features = 128, n_output = 1, padding = "CONSTANT"):
    inputs = keras.Input(shape = (input_shape[0], input_shape[1],n_output), dtype = tf.float32)
    inputs_2 = keras.Input(shape = (input_shape[0], input_shape[1],n_output), dtype = tf.float32)

    conv = resnet_block(inputs, n_base_features, kernel_size = 1, reps = 0, pooling = False, padding = padding)

    spade_1 = spade_generator_unit(conv,
                                   inputs_2,
                                   n_base_features,
                                   1,
                                   upsampling = False,
                                   padding = padding)
    
    conv_2 = resnet_block(spade_1, n_base_features, kernel_size = 1, reps = 2, pooling = False, padding = padding)
    conv4 = Conv2D(n_output, 1, padding = 'valid')(conv_2)
        
    integrator_resnet = keras.Model([inputs, inputs_2], conv4)
    return integrator_resnet
