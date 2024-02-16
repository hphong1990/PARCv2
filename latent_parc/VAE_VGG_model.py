import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, regularizers
from keras.layers import *

import os
os.chdir(".")
from parc import layer

class Binary2RGB(layers.Layer):

    def call(self, inputs):
        return tf.image.grayscale_to_rgb(inputs)
####
def vgg():
    inputs = keras.Input(shape = (160,256,1))
    rgb = Binary2RGB()(inputs)
    vgg = tf.keras.applications.inception_v3.InceptionV3(include_top=False,
                                                       weights='imagenet',
                                                       input_shape = (160,256,3),   
                                                       pooling=max)
    feature = vgg(rgb)
    vgg_model = keras.Model(inputs, feature)
    vgg_model.trainable = False
    return vgg_model
####
class Sampling(layers.Layer):
    """Uses (z_mean, z_log_var) to sample z, the vector encoding a digit."""

    def call(self, inputs):
        z_mean, z_log_var = inputs

        epsilon = tf.random.normal(shape=tf.shape(z_mean))
        return z_mean + tf.exp(0.5 * z_log_var) * epsilon
####
def vgg_encoder(latent_dims = 4, input_shape = (160,256,1), n_base_features = 64):
    inputs = keras.Input(shape = input_shape)
    conv1 = layer.conv_block_down(inputs,
                            feat_dim = n_base_features,
                            reps = 1,
                            kernel_size = 3)
                            # mode = 'down')
    conv2 = layer.conv_block_down(conv1,
                            feat_dim = n_base_features*2,
                            reps = 1,
                            kernel_size = 3,
                            mode = 'down')
    conv3 = layer.conv_block_down(conv2,
                            feat_dim = n_base_features*4,
                            reps = 2,
                            kernel_size = 3,
                            mode = 'down')
    conv4 = layer.conv_block_down(conv3,
                            feat_dim = n_base_features*8,
                            reps = 2,
                            kernel_size = 3,
                            mode = 'down')
    # conv5 = layer.conv_block_down(conv4,
    #                         feat_dim = n_base_features*8,
    #                         reps = 2,
    #                         kernel_size = 3,
    #                         mode = 'down')   
    
    z_mean = layers.Conv2D(latent_dims,3, padding="same",name="z_mean")(conv4)
    z_log_var = layers.Conv2D(latent_dims,3, padding="same",name="z_log_var")(conv4)
    z = Sampling()([z_mean,z_log_var])
    encoder = keras.Model(inputs, [z_mean,z_log_var,z])
    return encoder
# vgg_encoder().summary()
def vgg_decoder(input_shape = (20,32,4), n_base_features = 64):
    inputs = keras.Input(shape = input_shape)
    conv_in = layers.Conv2D(n_base_features*8, 3, activation = LeakyReLU(0.2), padding="same")(inputs)

    # conv1 = layer.conv_block_up_wo_concat(conv_in,
    #                         feat_dim = n_base_features*8,
    #                         reps = 2,
    #                         kernel_size = 3,
    #                         mode = 'up')
    conv2 = layer.conv_block_up_wo_concat(conv_in,
                            feat_dim = n_base_features*8,
                            reps = 2,
                            kernel_size = 3,
                            mode = 'up')
    conv3 = layer.conv_block_up_wo_concat(conv2,
                            feat_dim = n_base_features*4,
                            reps = 1,
                            kernel_size = 3,
                            mode = 'up')
    conv4 = layer.conv_block_up_wo_concat(conv3,
                            feat_dim = n_base_features*2,
                            reps = 1,
                            kernel_size = 3,
                            mode = 'up')
    conv5 = layer.conv_block_up_wo_concat(conv4,
                            feat_dim = n_base_features,
                            reps = 1,
                            kernel_size = 3)
    conv_out = layers.Conv2D(1, 3, padding="same")(conv5)
    decoder = keras.Model(inputs, conv_out)
    return decoder
# vgg_decoder().summary()
class VAE(keras.Model):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.encoder = vgg_encoder()
        self.decoder = vgg_decoder()
        self.total_loss_tracker = keras.metrics.Mean(name="total_loss")
        self.reconstruction_loss_tracker = keras.metrics.Mean(
            name="reconstruction_loss"
        )
        self.kl_loss_tracker = keras.metrics.Mean(name="kl_loss")
        self.vgg19 = vgg()
        
    @property
    def metrics(self):
        return [
            self.total_loss_tracker,
            self.reconstruction_loss_tracker,
            self.kl_loss_tracker,
        ]

    def compute_edge_boosting_weight(self, x_in):
        dy, dx = tf.image.image_gradients(x_in)
        G = tf.sqrt(dy**2+dx**2)
        normalized_g = (G - tf.reduce_min(G)) / (tf.reduce_max(G) - tf.reduce_min(G) + 1e-5)
        return normalized_g
        
    def edge_boosting_loss(self, y_pred, gt):
        # Compute the weight
        weight = self.compute_edge_boosting_weight(gt)

        # mse of weighted pixel val
        weighted_pred = tf.math.multiply(y_pred, weight)
        weighted_gt = tf.math.multiply(gt, weight)
        edge_boosting_mse = tf.keras.metrics.mean_absolute_error(weighted_gt, weighted_pred)

        # Sum all dimension
        edge_boosting_loss = tf.math.reduce_sum(edge_boosting_mse, axis = (0,1,2))
        return edge_boosting_loss
    
    def compute_hs_boosting_weight(self, x_in):
        hs_mask = tf.zeros(shape = tf.shape(x_in))
        hs_mask = tf.where(x_in < 0.11, tf.zeros_like(hs_mask), hs_mask)
        hs_mask = tf.where(x_in > 0.11, tf.ones_like(hs_mask), hs_mask)
        return hs_mask
        
    def hs_boosting_loss(self, y_pred, gt):
        # Compute the weight
        weight = self.compute_hs_boosting_weight(gt)

        # mse of weighted pixel val
        weighted_pred = tf.math.multiply(y_pred, weight)
        weighted_gt = tf.math.multiply(gt, weight)
        hs_boosting_mae = tf.keras.metrics.mean_absolute_error(weighted_gt, weighted_pred)

        # Sum all dimension
        hs_boosting_loss = tf.math.reduce_sum(hs_boosting_mae, axis = (0,1,2))
        return hs_boosting_loss
    def perceptual_loss(self, y_pred, gt):
        # Pred perceptaul
        pred_feature = self.vgg19(y_pred)
        # GT perceptual
        gt_feature = self.vgg19(gt)
        return tf.keras.losses.MeanSquaredError(reduction = 'sum')(pred_feature,gt_feature)
    
    def train_step(self, data):
        data = tf.cast(data,dtype = tf.float32)
        with tf.GradientTape() as tape:
            z_mean, z_log_var, z = self.encoder(data)
            reconstruction = self.decoder(z)
            reconstruction_loss = tf.keras.losses.MeanAbsoluteError(reduction = 'sum')(reconstruction,data)
            
            kl_loss = -0.5 * (1 + z_log_var - tf.square(z_mean) - tf.exp(z_log_var))
            kl_loss = (tf.reduce_sum(kl_loss, axis=(1,2,3)))
            hs_boost_loss = self.hs_boosting_loss(reconstruction,data)
            edge_boost_loss = self.edge_boosting_loss(reconstruction,data)
            perceptual_loss = self.perceptual_loss(reconstruction,data)
            total_loss = reconstruction_loss + kl_loss + hs_boost_loss + edge_boost_loss + perceptual_loss
            
        grads = tape.gradient(total_loss, self.trainable_weights)
        self.optimizer.apply_gradients(zip(grads, self.trainable_weights))
        self.total_loss_tracker.update_state(total_loss)
        self.reconstruction_loss_tracker.update_state(reconstruction_loss)
        self.kl_loss_tracker.update_state(kl_loss)
        return {
            "loss": self.total_loss_tracker.result(),
            "reconstruction_loss": self.reconstruction_loss_tracker.result(),
            "kl_loss": self.kl_loss_tracker.result(),
        }
    
    def predict(self, data):
        z_mean, z_log_var, z = self.encoder(data)
        reconstruction = self.decoder(z).numpy()
        concat = np.squeeze(np.stack([data, reconstruction], axis=0))  
        concat_rescale = (2 * concat) - 1
        return concat_rescale