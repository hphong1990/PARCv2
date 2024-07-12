from tensorflow import keras
from tensorflow.keras import  layers, regularizers
from keras.layers import *
import tensorflow as tf
import tensorflow_probability as tfp
from ...PARC import layer

from tensorflow.keras.layers import Concatenate, Input
from tensorflow.keras.models import Model

"""
Differentiator for EM problems: 
    - state vars including temperature, pressure, microstructure evolution
    - there is no constant field using
"""
tf.keras.backend.set_floatx('float32')


def differentiator_em(image_size, n_state_var=3):
    feature_extraction = layer.feature_extraction_resnet(input_shape=(image_size[0], image_size[1]), n_channel=n_state_var+2)
    
    mapping_and_recon = []
    mapping_and_recon.append(layer.mapping_and_recon_cnn(input_shape=(image_size[0], image_size[1]), n_mask_channel=2, output_channel=1))
    mapping_and_recon.append(layer.mapping_and_recon_cnn(input_shape=(image_size[0], image_size[1]), n_mask_channel=1, output_channel=1))
    mapping_and_recon.append(layer.mapping_and_recon_cnn(input_shape=(image_size[0], image_size[1]), n_mask_channel=1, output_channel=1))
    
    advection = [layer.Advection() for _ in range(n_state_var+2)]
    diffusion = layer.Diffusion()
    velocity_mapping_and_recon = layer.mapping_and_recon_cnn(input_shape=(image_size[0], image_size[1]), n_mask_channel=2, output_channel=2)

    # Main computation graph
    input_tensor = Input(shape=(image_size[0] , image_size[1], n_state_var+2), dtype=tf.float32)
    init_state_var = input_tensor[:,:,:,:n_state_var]
    velocity_field = input_tensor[:,:,:,n_state_var:]

    # Reaction term
    dynamic_feature = feature_extraction(input_tensor)

    # Temp
    advec_temp = advection[0](init_state_var[:, :, :, 0:1], velocity_field)
    diffusion_temp = diffusion(init_state_var[:, :, :, 0:1])
    temp_concat = Concatenate(axis=-1)([advec_temp,diffusion_temp])
    temp_dot = mapping_and_recon[0]([dynamic_feature, temp_concat])
    
    # Pressure
    advec_press = advection[1](init_state_var[:, :, :, 1:2], velocity_field)
    press_dot = mapping_and_recon[1]([dynamic_feature, advec_press])
    
    # Micro
    advec_micro = advection[2](init_state_var[:, :, :, 2:3], velocity_field)
    micro_dot = mapping_and_recon[2]([dynamic_feature, advec_micro])
    
    # Velocity
    advec_vel = []
    for i in range(2):
        advec_i = advection[i+3](velocity_field[:, :, :, i:i+1], velocity_field)
        advec_vel.append(advec_i)
        
    advec_vel_concat = Concatenate(axis=-1)(advec_vel)
    velocity_dot = velocity_mapping_and_recon([dynamic_feature, advec_vel_concat])
    output_tensor = Concatenate(axis=-1)([temp_dot,press_dot,micro_dot, velocity_dot])
    
    differentiator = Model(input_tensor, output_tensor)
    return differentiator


class PARCv2(keras.Model):
    def __init__(self, n_state_var, n_time_step, step_size, image_size=(0, 0), int_rtol=1e-3, int_atol=1e-6, **kwargs):
        super(PARCv2, self).__init__(**kwargs)
        self.n_state_var = n_state_var
        self.n_time_step = n_time_step
        self.step_size = step_size
        self.t_eval = tf.linspace(1, n_time_step, n_time_step) * step_size
        
        self.differentiator = differentiator_em(n_state_var=self.n_state_var, image_size=image_size)
        self.integrator = tfp.math.ode.DormandPrince(rtol=int_rtol, atol=int_atol)
        self.total_loss_tracker = keras.metrics.Mean(name="total_loss")

    @property
    def metrics(self):
        return [
        self.total_loss_tracker,
        ]
    
    def _int_diff_call(self, t, y):
        return tf.cast(self.differentiator(y), dtype=tf.float32)
    
    @tf.function
    def call(self, input):
        state_var_init = tf.cast(input[0], dtype=tf.float32)
        velocity_init = tf.cast(input[1], dtype=tf.float32)
        ic = Concatenate(axis = -1)([state_var_init, velocity_init])
        res = self.integrator.solve(self._int_diff_call, 0.0, ic, solution_times=self.t_eval)
        return res

    @tf.function
    def train_step(self, data):
        state_var_init = tf.cast(data[0][0],dtype = tf.float32)
        velocity_init = tf.cast(data[0][1], dtype = tf.float32)
        input_seq = Concatenate(axis = -1)([state_var_init, velocity_init])

        state_var_gt = tf.cast(data[1][0], dtype = tf.float32)
        velocity_gt = tf.cast(data[1][1], dtype = tf.float32)

        with tf.GradientTape() as tape:
            tape.watch(self.differentiator.trainable_weights)
            results = self.integrator.solve(self._int_diff_call, 0.0, input_seq, solution_times=self.t_eval)
            state_pred, vel_pred = results.states[:, :, :, :, :self.n_state_var], results.states[:, :, :, :, self.n_state_var:]
            total_loss  = (tf.keras.losses.MeanAbsoluteError(reduction = 'sum')(state_pred, state_var_gt) + 
                           tf.keras.losses.MeanAbsoluteError(reduction = 'sum')(vel_pred, velocity_gt)) / 2
            
        grads = tape.gradient(total_loss, self.differentiator.trainable_weights)
        self.optimizer.apply_gradients(zip(grads, self.differentiator.trainable_weights))
        self.total_loss_tracker.update_state(total_loss)

        return {
            "total_loss": self.total_loss_tracker.result(),
        }
