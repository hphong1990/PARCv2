from tensorflow import keras
from tensorflow.keras import  layers, regularizers
from keras.layers import *
import tensorflow as tf
from parc import layer

from tensorflow.keras.layers import Concatenate, Input
from tensorflow.keras.models import Model

"""
Model definition customized for Burgers' problems: 
    - In this problem, state vars and velocity only include velocity
    - Architecture was adjusted to make it lighter and comparable with PhyCRNet
"""

def differentiator_burgers():
    # Model initiation
    feature_extraction = layer.feature_extraction_burgers(input_shape = (64,64), n_channel = 3)
    
    advection = [layer.Advection() for _ in range(2)]
    diffusion = [layer.Diffusion() for _ in range(2)]
    velocity_mapping_and_recon = layer.mapping_and_recon_cnn(input_shape = (64,64), n_base_features = 64, n_mask_channel=4, output_channel=2)

    # Main computation graph
    velocity_field = Input(shape=(64 , 64, 3), dtype = tf.float32)

    # Reaction term
    dynamic_feature = feature_extraction(velocity_field)

    # u
    advec_u = advection[0](velocity_field[:, :, :, 0:1], velocity_field[:, :, :, 0:2])
    diffusion_u = diffusion[0](velocity_field[:, :, :, 0:1])
    
    # v
    advec_v = advection[1](velocity_field[:, :, :, 1:2], velocity_field[:, :, :, 0:2])
    diffusion_v = diffusion[1](velocity_field[:, :, :, 1:2])    
    
    # Concatenate
    advec_diff_concat = Concatenate(axis=-1)([advec_u,advec_v,diffusion_u,diffusion_v])
    
    # Final mapping
    velocity_dot = velocity_mapping_and_recon([dynamic_feature, advec_diff_concat])
    
    differentiator = Model(velocity_field, velocity_dot)
    return differentiator

def integrator_burgers():
    velocity_integrator = layer.integrator_cnn(input_shape = (64,64), n_base_features = 64, n_output=2)

    velocity_prev = keras.layers.Input(shape = (64, 64, 2), dtype = tf.float32)
    velocity_dot = keras.layers.Input(shape = (64, 64, 2), dtype = tf.float32)

    velocity_next = velocity_integrator([velocity_dot, velocity_prev])
    integrator = keras.Model([velocity_dot, velocity_prev], [velocity_next])
    return integrator

class PARCv2_burgers(keras.Model):
    def __init__(self, n_time_step, step_size, solver = "rk4", mode = "integrator_training", use_data_driven_int = True, differentiator_backbone = 'em', **kwargs):
        super(PARCv2_burgers, self).__init__(**kwargs)
        self.n_time_step = n_time_step
        self.step_size = step_size
        self.solver = solver
        self.mode = mode
        self.use_data_driven_int = use_data_driven_int
        self.differentiator = differentiator_burgers()
        self.integrator = integrator_burgers()
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
        velocity_init = tf.cast(data[0], dtype = tf.float32)
        velocity_gt = tf.cast(data[1], dtype = tf.float32)

        input_seq_current = velocity_init
        with tf.GradientTape() as tape:
            if self.mode == "integrator_training":
                for ts in range(self.n_time_step):
                    # Compute k1
                    input_seq_current, update = self.explicit_update(input_seq_current)
                    velocity_next = self.integrator([update, input_seq_current])
                    input_seq_current = Concatenate(axis = -1)([velocity_next,velocity_init[:,:,:,2:]])
                
            else: 
                velocity_next, update = self.explicit_update(input_seq_current)

            total_loss  = tf.keras.losses.MeanAbsoluteError(reduction = 'sum')(input_seq_current,velocity_gt)
                           
        grads = tape.gradient(total_loss, self.trainable_weights)
        self.optimizer.apply_gradients(zip(grads, self.trainable_weights))

        self.total_loss_tracker.update_state(total_loss)

        return {
            "total_loss": self.total_loss_tracker.result(),
        }
    
    # Update scheme
    def explicit_update(self, input_seq_current):
        input_seq_current = tf.clip_by_value(input_seq_current, 0, 1)

        if self.solver == "rk4":
            input_seq_current, update = self.rk4_update(input_seq_current)
        elif self.solver == 'heun':
            input_seq_current, update = self.heun_update(input_seq_current)
        else:
            input_seq_current, update = self.euler_update(input_seq_current)

        return input_seq_current, update

    def rk4_update(self, input_seq_current):

        # Compute k1
        k1 = self.differentiator(input_seq_current)

        # Compute k2
        inp_k2 = input_seq_current[:,:,:,:2] + self.step_size*1/2*k1 
        inp_k2 = Concatenate(axis = -1)([inp_k2,input_seq_current[:,:,:,2:]])

        k2 = self.differentiator(inp_k2)

        # Compute k3
        inp_k3 = input_seq_current[:,:,:,:2] + self.step_size*1/2*k2
        inp_k3 = Concatenate(axis = -1)([inp_k3,input_seq_current[:,:,:,2:]])
        k3 = self.differentiator(inp_k3)

        # Compute k4
        inp_k4 = input_seq_current[:,:,:,:2] + self.step_size*k3
        inp_k4 = Concatenate(axis = -1)([inp_k4,input_seq_current[:,:,:,2:]])

        k4 = self.differentiator(inp_k4)

        # Final
        update = 1/6*(k1 + 2*k2 + 2*k3 + k4)
        final_state = input_seq_current[:,:,:,:2] + self.step_size*update 
        input_seq_current = Concatenate(axis = -1)([final_state,input_seq_current[:,:,:,2:]])
        return input_seq_current, update
    
    # Euler update function
    def heun_update(self, input_seq_current):
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
        # Compute update
        update = self.differentiator(input_seq_current)
        input_seq_current = input_seq_current + self.step_size*update 

        return input_seq_current, update