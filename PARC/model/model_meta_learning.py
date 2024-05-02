from tensorflow import keras
from tensorflow.keras import  layers, regularizers
from keras.layers import *
import tensorflow as tf
from PARC import layer

from tensorflow.keras.layers import Concatenate, Input
from tensorflow.keras.models import Model

"""
Differentiator for EM problems: 
    - state vars including temperature, pressure, microstructure evolution
    - there is no constant field using
"""

def differentiator_em(image_size, n_state_var=3):
    # Model initiation
    ## changing to  128 and 216 given ratio 600 x 1000
    feature_extraction = layer.feature_extraction_unet(input_shape = (image_size[0], image_size[1]), n_channel=n_state_var+2)
    
    mapping_and_recon = []
    mapping_and_recon.append(layer.mapping_and_recon_cnn(input_shape = (image_size[0], image_size[1]), n_mask_channel=2, output_channel=1))
    mapping_and_recon.append(layer.mapping_and_recon_cnn(input_shape = (image_size[0], image_size[1]), n_mask_channel=1, output_channel=1))
    mapping_and_recon.append(layer.mapping_and_recon_cnn(input_shape = (image_size[0], image_size[1]), n_mask_channel=1, output_channel=1))
    
    advection = [layer.Advection() for _ in range(n_state_var+2)]
    diffusion = layer.Diffusion()
    velocity_mapping_and_recon = layer.mapping_and_recon_cnn(input_shape = (image_size[0], image_size[1]), n_mask_channel=2, output_channel=2)

    # Main computation graph
    input_tensor = Input(shape=(image_size[0] , 208, n_state_var+2), dtype = tf.float32)
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

def integrator( image_size, n_state_var = 3):
    state_integrators = []
    for _ in range(n_state_var):
        state_integrators.append(layer.integrator_cnn(input_shape = (image_size[0],image_size[1])))

    velocity_integrator = layer.integrator_cnn(input_shape = (image_size[0],image_size[1]), n_output=2)

    state_var_prev = keras.layers.Input(shape = (image_size[0], image_size[1], n_state_var), dtype = tf.float32)
    velocity_prev = keras.layers.Input(shape = (image_size[0], image_size[1],2), dtype = tf.float32)
    
    state_var_dot = keras.layers.Input(shape = (image_size[0], image_size[1],n_state_var), dtype = tf.float32)
    velocity_dot = keras.layers.Input(shape = (image_size[0], image_size[1],2), dtype = tf.float32)

    state_var_next = []
        
    for i in range(n_state_var): 
        state_var_next.append(state_integrators[i]([state_var_dot[:,:,:,i:i+1], state_var_prev[:,:,:,i:i+1]]))

    state_var_next = keras.layers.concatenate(state_var_next, axis=-1)
    velocity_next = velocity_integrator([velocity_dot, velocity_prev])
    integrator = keras.Model([state_var_dot, velocity_dot, state_var_prev, velocity_prev], [state_var_next, velocity_next])
    return integrator

class PARCv2(keras.Model):
    def __init__(self, n_state_var, n_time_step, step_size, solver = "rk4", mode = "integrator_training", use_data_driven_int = True, differentiator_backbone = 'em', image_size = (0, 0), **kwargs):
        super(PARCv2, self).__init__(**kwargs)
        self.n_state_var = n_state_var
        self.n_time_step = n_time_step
        self.step_size = step_size
        self.solver = solver
        self.mode = mode
        self.use_data_driven_int = use_data_driven_int
        
        self.differentiator = differentiator_em(n_state_var=self.n_state_var, image_size = image_size)
        self.integrator = integrator(image_size = image_size)
        self.total_loss_tracker = keras.metrics.Mean(name="total_loss")
        if self.mode == "integrator_training":
            self.differentiator.trainable = False
        else:
            self.integrator.trainable = False

    @property
    def metrics(self):
        return [
        self.total_loss_tracker,
        ]
    
    def call(self, input):
        state_var_init = tf.cast(input[0],dtype = tf.float32)
        velocity_init = tf.cast(input[1], dtype = tf.float32)
        input_seq = Concatenate(axis = -1)([state_var_init, velocity_init])

        input_seq_current = input_seq

        res = []
        for _ in range(self.n_time_step):    
            input_seq_current, update = self.explicit_update(input_seq_current)
            if self.use_data_driven_int == True:
                state_var_next, velocity_next = self.integrator([update[:,:,:,:3],update[:,:,:,3:],input_seq_current[:,:,:,:3], input_seq_current[:,:,:,3:]])
                input_seq_current = Concatenate()([state_var_next, velocity_next])
                        
            res.append(input_seq_current)
        return res

    @tf.function
    def train_step(self, data):
        state_var_init = tf.cast(data[0][0],dtype = tf.float32)
        velocity_init = tf.cast(data[0][1], dtype = tf.float32)
        input_seq = Concatenate(axis = -1)([state_var_init, velocity_init])

        state_var_gt = tf.cast(data[1][0], dtype = tf.float32)
        velocity_gt = tf.cast(data[1][1], dtype = tf.float32)

        input_seq_current = input_seq
        with tf.GradientTape() as tape:
            state_whole = []
            vel_whole = []
            if self.mode == "integrator_training":
                for ts in range(self.n_time_step):
                    # Compute k1
                    input_seq_current, update = self.explicit_update(input_seq_current)
                    state_var_next, velocity_next = self.integrator([update[:,:,:,:3],update[:,:,:,3:],input_seq_current[:,:,:,:3], input_seq_current[:,:,:,3:]])
                    input_seq_current = Concatenate()([state_var_next, velocity_next])
                    state_whole.append(state_var_next)
                    vel_whole.append(velocity_next)
            else: 
                for ts in range(self.n_time_step):
                    input_seq_current, update = self.explicit_update(input_seq_current)
                    state_whole.append( input_seq_current[:,:,:,:3])
                    vel_whole.append(input_seq_current[:,:,:,3:])
            state_pred = Concatenate(axis = -1)(state_whole)
            vel_pred = Concatenate(axis = -1)(vel_whole)
                    
            total_loss  = (tf.keras.losses.MeanAbsoluteError(reduction = 'sum')(state_pred,state_var_gt) + 
                            tf.keras.losses.MeanAbsoluteError(reduction = 'sum')(vel_pred,velocity_gt))/2
        grads = tape.gradient(total_loss, self.trainable_weights)
        self.optimizer.apply_gradients(zip(grads, self.trainable_weights))

        self.total_loss_tracker.update_state(total_loss)

        return {
            "total_loss": self.total_loss_tracker.result(),
        }
    
    # Update scheme
    def explicit_update(self, input_seq_current):
        if self.solver == "rk4":
            input_seq_current, update = self.rk4_update(input_seq_current)
        elif self.solver == 'heun':
            input_seq_current, update = self.heun_update(input_seq_current)
        else:
            input_seq_current, update = self.euler_update(input_seq_current)

        return input_seq_current, update

    def rk4_update(self, input_seq_current):
        input_seq_current = tf.clip_by_value(input_seq_current, 0, 1)

        # Compute k1
        k1 = self.differentiator(input_seq_current)

        # Compute k2
        inp_k2 = input_seq_current + self.step_size*1/2*k1 
        k2 = self.differentiator(inp_k2)

        # Compute k3
        inp_k3 = input_seq_current + self.step_size*1/2*k2
        k3 = self.differentiator(inp_k3)

        # Compute k4
        inp_k4 = input_seq_current + self.step_size*k3
        k4 = self.differentiator(inp_k4)

        # Final
        update = 1/6*(k1 + 2*k2 + 2*k3 + k4)
        input_seq_current = input_seq_current + self.step_size*update 
        return input_seq_current, update
    
    # Euler update function
    def heun_update(self, input_seq_current):
        input_seq_current = tf.clip_by_value(input_seq_current, 0, 1)
        # Compute update
        k1 = self.differentiator(input_seq_current)

        # Compute k2
        inp_k2 = input_seq_current + self.step_size*k1 
        k2 = self.differentiator(inp_k2)
        
        update = 1/2*(k1 + k2)
        input_seq_current = input_seq_current + self.step_size*update 

        return input_seq_current, update
    
    # Euler update function
    def euler_update(self, input_seq_current):
        input_seq_current = tf.clip_by_value(input_seq_current, 0, 1)
        # Compute update
        update = self.differentiator(input_seq_current)
        input_seq_current = input_seq_current + self.step_size*update 