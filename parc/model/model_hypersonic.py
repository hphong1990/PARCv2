from tensorflow import keras
from tensorflow.keras import  layers, regularizers
from keras.layers import *
import tensorflow as tf
from PARCv2.parc import layer

from tensorflow.keras.layers import Concatenate, Input
from tensorflow.keras.models import Model

"""
Differentiator for hypersonic flow problem: 
    - state vars: density, pressure
    - there is no constant field using
"""
DATA_SHAPE = (112, 176)

def differentiator_em(n_state_var=2):
    # Model initiation
    feature_extraction = layer.feature_extraction_unet(input_shape=DATA_SHAPE,
                                                       n_channel=n_state_var+2)
    
    mapping_and_recon = []
    # rho_dot: n_mask_channel=2 because of advec+diff
    mapping_and_recon.append(layer.mapping_and_recon_cnn(input_shape=DATA_SHAPE,
                                                         n_mask_channel=2, 
                                                         output_channel=1))
    # p_dot: n_mask_channel=1 because of no diffusion
    mapping_and_recon.append(layer.mapping_and_recon_cnn(input_shape=DATA_SHAPE,
                                                         n_mask_channel=1,
                                                         output_channel=1))
    
    advection = [layer.Advection() for _ in range(n_state_var+2)]
    diffusion = layer.Diffusion()
    velocity_mapping_and_recon = layer.mapping_and_recon_cnn(input_shape=DATA_SHAPE,
                                                             n_mask_channel=2,
                                                             output_channel=2)

    # Main computation graph
    input_tensor = Input(shape=(DATA_SHAPE[0], DATA_SHAPE[1], n_state_var+2), 
                         dtype = tf.float32)
    init_state_var = input_tensor[:,:,:,:n_state_var]
    velocity_field = input_tensor[:,:,:,n_state_var:]

    # Reaction term
    dynamic_feature = feature_extraction(input_tensor)

    # Density
    advec_rho = advection[0](init_state_var[:, :, :, 0:1], velocity_field)
    diffusion_rho = diffusion(init_state_var[:, :, :, 0:1])
    rho_concat = Concatenate(axis=-1)([advec_rho, diffusion_rho])
    rho_dot = mapping_and_recon[0]([dynamic_feature, rho_concat])
    
    # Pressure
    advec_press = advection[1](init_state_var[:, :, :, 1:2], velocity_field)
    press_dot = mapping_and_recon[1]([dynamic_feature, advec_press])
    
    # Velocity
    advec_vel = []
    for i in range(2):
        advec_i = advection[i+2](velocity_field[:, :, :, i:i+1], velocity_field)
        advec_vel.append(advec_i)
        
    advec_vel_concat = Concatenate(axis=-1)(advec_vel)
    velocity_dot = velocity_mapping_and_recon([dynamic_feature, advec_vel_concat])
    
    
    output_tensor = Concatenate(axis=-1)([rho_dot, press_dot, velocity_dot])
    differentiator = Model(input_tensor, output_tensor)
    return differentiator

def integrator(n_state_var=2):
    state_integrators = []
    for _ in range(n_state_var):
        state_integrators.append(layer.integrator_cnn(input_shape=DATA_SHAPE))

    velocity_integrator = layer.integrator_cnn(input_shape=DATA_SHAPE, n_output=2)

    state_var_prev = keras.layers.Input(shape=(DATA_SHAPE[0], DATA_SHAPE[1], n_state_var), 
                                        dtype=tf.float32)
    velocity_prev = keras.layers.Input(shape=(DATA_SHAPE[0], DATA_SHAPE[1], 2),
                                       dtype=tf.float32)
    
    state_var_dot = keras.layers.Input(shape=(DATA_SHAPE[0], DATA_SHAPE[1], n_state_var), 
                                       dtype=tf.float32)
    velocity_dot = keras.layers.Input(shape=(DATA_SHAPE[0], DATA_SHAPE[1], 2), dtype=tf.float32)

    state_var_next = []
        
    for i in range(n_state_var): 
        state_var_next.append(state_integrators[i]([state_var_dot[:,:,:,i:i+1],
                                                    state_var_prev[:,:,:,i:i+1]]))

    state_var_next = keras.layers.concatenate(state_var_next, axis=-1)
    velocity_next = velocity_integrator([velocity_dot, velocity_prev])
    integrator = keras.Model([state_var_dot, velocity_dot, state_var_prev, velocity_prev],
                             [state_var_next, velocity_next])
    return integrator

@keras.saving.register_keras_serializable()
class PARCv2(keras.Model):
    def __init__(self, n_state_var, n_time_step, step_size, solver="rk4",
                 mode="integrator_training", use_data_driven_int=True,
                 differentiator_backbone="em", **kwargs):
        super(PARCv2, self).__init__(**kwargs)
        self.n_state_var = n_state_var
        self.n_time_step = n_time_step
        self.step_size = step_size
        self.solver = solver
        self.mode = mode
        self.use_data_driven_int = use_data_driven_int
        
        self.differentiator = differentiator_em(n_state_var=self.n_state_var)
        self.integrator = integrator()
        self.total_loss_tracker = keras.metrics.Mean(name="total_loss")
        if self.mode == "integrator_training":
            self.differentiator.trainable = False
        else:
            self.integrator.trainable = False
        
        self.input_layer1 = keras.layers.Input((DATA_SHAPE[0], DATA_SHAPE[1], 2))
        self.input_layer2 = keras.layers.Input((DATA_SHAPE[0], DATA_SHAPE[1], 2))
        self.out = self.call([self.input_layer1, self.input_layer2])
        
        super(PARCv2, self).__init__(
            inputs=[self.input_layer1, self.input_layer2], outputs=self.out, **kwargs
        )

    def build(self):
        self._is_graph_network = True
        self._init_graph_network(
            inputs=[self.input_layer1, self.input_layer2], outputs=self.out
        )

    @property
    def metrics(self):
        return [
        self.total_loss_tracker,
        ]
    
    def call(self, input, training=False):
        state_var_init = tf.cast(input[0],dtype = tf.float32)
        velocity_init = tf.cast(input[1], dtype = tf.float32)
        input_seq = Concatenate(axis = -1)([state_var_init, velocity_init])

        input_seq_current = input_seq

        res = []
        for _ in range(self.n_time_step):    
            input_seq_current, update = self.explicit_update(input_seq_current)
            if self.use_data_driven_int == True:
                state_var_next, velocity_next = self.integrator([update[:,:,:,:2],
                                                                 update[:,:,:,2:],
                                                                 input_seq_current[:,:,:,:2],
                                                                 input_seq_current[:,:,:,2:]])
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
            for ts in range(self.n_time_step):
                # Compute k1
                input_seq_current, update = self.explicit_update(input_seq_current)
                if self.mode == "integrator_training":
                    state_var_next, velocity_next = self.integrator([update[:,:,:,:2],
                                                                     update[:,:,:,2:], input_seq_current[:,:,:,:2], input_seq_current[:,:,:,2:]])
                    input_seq_current = Concatenate()([state_var_next, velocity_next])

            state_var_next = input_seq_current[:,:,:,:2]
            velocity_next = input_seq_current[:,:,:,2:]
                    
            total_loss  = (tf.keras.losses.MeanAbsoluteError(reduction = 'sum')(state_var_next,state_var_gt) + 
                            tf.keras.losses.MeanAbsoluteError(reduction = 'sum')(velocity_next,velocity_gt))/2
        grads = tape.gradient(total_loss, self.trainable_weights)
        self.optimizer.apply_gradients(zip(grads, self.trainable_weights))

        self.total_loss_tracker.update_state(total_loss)

        return {
            "total_loss": self.total_loss_tracker.result(),
        }

    @tf.function
    def test_step(self, data):
        state_var_init = tf.cast(data[0][0],dtype = tf.float32)
        velocity_init = tf.cast(data[0][1], dtype = tf.float32)
        input_seq = Concatenate(axis = -1)([state_var_init, velocity_init])

        state_var_gt = tf.cast(data[1][0], dtype = tf.float32)
        velocity_gt = tf.cast(data[1][1], dtype = tf.float32)

        input_seq_current = input_seq
        for ts in range(self.n_time_step):
            input_seq_current, update = self.explicit_update(input_seq_current)
            if self.mode == "integrator_training":
                state_var_next, velocity_next = self.integrator([update[:,:,:,:2], update[:,:,:,2:], 
                                                                 input_seq_current[:,:,:,:2], input_seq_current[:,:,:,2:]])
                input_seq_current = Concatenate()([state_var_next, velocity_next])
                
        state_var_next = input_seq_current[:,:,:,:2]
        velocity_next = input_seq_current[:,:,:,2:]
                    
        total_loss  = (tf.keras.losses.MeanAbsoluteError(reduction = 'sum')(state_var_next,state_var_gt) + 
                       tf.keras.losses.MeanAbsoluteError(reduction = 'sum')(velocity_next,velocity_gt))/2

        self.total_loss_tracker.update_state(total_loss)

        return {
            "val_loss": self.total_loss_tracker.result(),
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

        return input_seq_current, update
