from tensorflow import keras
from tensorflow.keras import  layers, regularizers
from keras.layers import *
import tensorflow as tf

def resnet_unit(feat_dim, kernel_size, x_in):
    """
    Resnet unit: x_in --> conv --> relu --> conv --> + relu --> x_out
                    |                           |
                     ---------------------------
    Parameter: 
                - feat_dim (int): number of channels
                - kernel_size (int): size of convolution kernel
                - x_in (tensor): input tensor
    Return:
                - (tensor): output tensor of residual unit.
    """
    res = keras.Sequential([
        Conv2D(feat_dim,
               kernel_size, padding="same"),
        ReLU(),
        Conv2D(feat_dim,
               kernel_size,
               padding="same")
    ])
    return ReLU()(x_in + res(x_in))

def resnet_block(x_in, feat_dim, kernel_size, reps, pooling = True):
    """
    Assembly of multiple resnet unit: x_in --> 2 x (conv + relu) --> 'reps' x resnet_unit --> output
    Parameter: 
                - x_in (tensor): input tensor
                - feat_dim (int): number of channels
                - kernel_size (int): size of convolution kernel
                - reps (int): number of residual unit
                - pooling (bool): the block ends with pooling or not
    Return:
                - (tensor): output of the resnet block
    """
    conv1 = Conv2D(feat_dim,
                   kernel_size,
                   padding="same")(x_in)
    relu1 = ReLU()(conv1)
    conv2 = Conv2D(feat_dim,
                   kernel_size,
                   padding="same")(relu1)
    x = ReLU()(conv2)
    for _ in range(reps):
        x = resnet_unit(feat_dim,kernel_size,x)
    if pooling == True:
        x = MaxPooling2D(2,2)(x)
        return x
    else:
        return x

def conv_unit(feat_dim, kernel_size, x):
    x = Conv2D(feat_dim, 
               kernel_size, 
               activation = LeakyReLU(0.2), 
               padding="same")(x)
    x = Conv2D(feat_dim,
               1,
               activation = LeakyReLU(0.2),
               padding="same")(x)
    # x = LayerNormalization()(x)
    return x

def conv_block_down(x, feat_dim, reps, kernel_size, mode = 'normal'):
    if mode == 'down':
        x = MaxPooling2D(2,2)(x)
    for _ in range(reps):
        x = conv_unit(feat_dim, 
                      kernel_size,
                      x)
    return x

def conv_block_up_w_concat(x, x1, feat_dim, reps, kernel_size, mode = 'normal'):
    if mode == 'up':
        x = UpSampling2D((2,2),interpolation='bilinear')(x)
    
    x = Concatenate()([x,x1])
    for _ in range(reps):
        x = conv_unit(feat_dim,
                      kernel_size,
                      x)
    return x

def conv_block_up_wo_concat(x, feat_dim, reps, kernel_size, mode = 'normal'):
    if mode == 'up':
        x = UpSampling2D((2,2),interpolation='bilinear')(x)
    for _ in range(reps):
        x = conv_unit(feat_dim,
                      kernel_size,
                      x)
    return x

class SPADE(layers.Layer):
    def __init__(self, filters, epsilon=1e-5, **kwargs):
        super().__init__(**kwargs)
        self.epsilon = epsilon
        self.conv = layers.Conv2D(filters, 3, padding="same", activation="relu")
        self.conv_gamma = layers.Conv2D(filters, 3, padding="same")
        self.conv_beta = layers.Conv2D(filters, 3, padding="same")

    def build(self, input_shape):
        self.resize_shape = input_shape[1:3]
        # print(self.resize_shape)

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

def feature_extraction_unet(input_shape = (128,192), n_out_features = 128, n_base_features = 64, n_channel = 5):
    inputs = keras.Input(shape = (input_shape[0], input_shape[1],n_channel))

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
    conv4 = conv_block_down(conv3,
                            feat_dim = n_base_features*8,
                            reps = 1,
                            kernel_size = 3,
                            mode = 'down')
    conv5 = conv_block_down(conv4,
                            feat_dim = n_base_features*16,
                            reps = 1,
                            kernel_size = 3,
                            mode = 'down')
    conv6 = conv_block_up_wo_concat(conv5,
                                    feat_dim = n_base_features*8,
                                    reps = 1,
                                    kernel_size = 3,
                                    mode = 'up')
    conv7 = conv_block_up_w_concat(conv6, conv3,
                                    feat_dim = n_base_features*4,
                                    reps = 1,
                                    kernel_size = 3,
                                    mode = 'up')
    conv8 = conv_block_up_wo_concat(conv7,
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



class Advection(layers.Layer):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def call(self, state_variable, velocity_field):
        dy, dx = tf.image.image_gradients(state_variable)
        spatial_deriv = tf.concat([dy,dx],axis = -1)
        advect = tf.reduce_sum(tf.multiply(spatial_deriv,velocity_field),axis = -1, keepdims=True)
        return advect
    
class Gradient(layers.Layer):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def call(self, state_variable):
        dy, dx = tf.image.image_gradients(state_variable)
        return dy, dx

class Diffusion(layers.Layer):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def call(self, state_variable):
        dy, dx = tf.image.image_gradients(state_variable)
        dyy, dyx = tf.image.image_gradients(dy)
        dxy, dxx = tf.image.image_gradients(dx)
        laplacian = tf.add(dyy,dxx)
        return laplacian
    
class Poisson(layers.Layer):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def call(self, vector_field):
        uy, ux = tf.image.image_gradients(vector_field[0])
        vy, vx = tf.image.image_gradients(vector_field[1])
        ux2 = tf.multiply(ux,ux)
        vy2 = tf.multiply(vy,vy)
        uyvx = tf.multiply(uy,vx)
        return ux2,vy2,uyvx

def mapping_and_recon_cnn(input_shape = (128,192), n_base_features = 128, n_mask_channel = 1, output_channel = 1 ):
    inputs = keras.Input(shape = (input_shape[0], input_shape[1], n_base_features), dtype = tf.float32)
    inputs_2 = keras.Input(shape = (input_shape[0], input_shape[1], n_mask_channel), dtype = tf.float32)
    
    # Style vector 
    spade_1 = spade_generator_unit(inputs,
                                   inputs_2,
                                   n_base_features,
                                   1,
                                   upsampling = False)
    conv_last = resnet_block(spade_1, n_base_features, kernel_size = 1, reps = 2, pooling = False)

    conv_out = Conv2D(output_channel,1, padding='same')(conv_last)
    mapping_resnet = keras.Model([inputs,inputs_2], conv_out)
    return mapping_resnet

def integrator_cnn(input_shape = (128,192), n_base_features = 128, n_output = 1):
    inputs = keras.Input(shape = (input_shape[0], input_shape[1],n_output), dtype = tf.float32)
    inputs_2 = keras.Input(shape = (input_shape[0], input_shape[1],n_output), dtype = tf.float32)

    conv = resnet_block(inputs, n_base_features, kernel_size = 1, reps = 0, pooling = False)

    spade_1 = spade_generator_unit(conv,
                                   inputs_2,
                                   n_base_features,
                                   1,
                                   upsampling = False)
    
    conv_2 = resnet_block(spade_1, n_base_features, kernel_size = 1, reps = 2, pooling = False)
    conv4 = Conv2D(n_output, 1, padding = 'same')(conv_2)
        
    integrator_resnet = keras.Model([inputs, inputs_2], conv4)
    return integrator_resnet