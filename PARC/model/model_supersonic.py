from tensorflow import keras
from tensorflow.keras import layers, regularizers
from keras.layers import *
import tensorflow as tf
from PARC import layer

from tensorflow.keras.layers import Concatenate, Input
from tensorflow.keras.models import Model

"""
Differentiator for hypersonic flow problem: 
    - state vars: density, pressure
    - there is no constant field
"""
DATA_SHAPE = (112, 176)

def differentiator_em(n_state_var=2, padding="SYMMETRIC"):
    # Model initiation
    feature_extraction = layer.feature_extraction_unet(input_shape=DATA_SHAPE,
                                                       n_channel=n_state_var+2,
                                                       padding=padding)
    
    mapping_and_recon = []
    # rho_dot: n_mask_channel=2 because of advec+diff
    mapping_and_recon.append(layer.mapping_and_recon_cnn(input_shape=DATA_SHAPE,
                                                         n_mask_channel=2, 
                                                         output_channel=1,
                                                         padding=padding))
    # p_dot: n_mask_channel=1 because of no diffusion
    mapping_and_recon.append(layer.mapping_and_recon_cnn(input_shape=DATA_SHAPE,
                                                         n_mask_channel=1,
                                                         output_channel=1,
                                                         padding=padding))
    
    advection = [layer.Advection() for _ in range(n_state_var+2)]
    diffusion = layer.Diffusion()
    velocity_mapping_and_recon = layer.mapping_and_recon_cnn(input_shape=DATA_SHAPE,
                                                             n_mask_channel=2,
                                                             output_channel=2,
                                                             padding=padding)

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


def integrator(n_state_var=2, padding="SYMMETRIC"):
    state_var_prev = keras.layers.Input(shape=(DATA_SHAPE[0], DATA_SHAPE[1], n_state_var+2), 
                                        dtype=tf.float32)
    state_var_dot = keras.layers.Input(shape=(DATA_SHAPE[0], DATA_SHAPE[1], n_state_var+2), 
                                       dtype=tf.float32)
    conv = layer.conv_block_down(state_var_dot, feat_dim=64, reps=1, kernel_size=5,
                                 mode='down', padding=padding)
    conv2 = layer.conv_block_down(conv, feat_dim=128, reps=1, kernel_size=5, 
                                  mode='normal', padding=padding)
    conv3 = layer.conv_block_up_wo_concat(conv2, feat_dim=64, reps=1, kernel_size=5,
                                          mode='up', padding=padding)
    conv3 = tf.pad(conv3, tf.constant([[0, 0], [1, 1], [1, 1], [0, 0]]), padding)
    conv_out = Conv2D(n_state_var+2, 3, padding='valid')(conv3)
    state_var_next = Add()([state_var_prev, conv_out])
    integrator = keras.Model([state_var_dot, state_var_prev], [state_var_next], 
                             name='integrator')
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
                input_seq_current = self.integrator([update, input_seq_current])
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
            res = []
            for ts in range(self.n_time_step):
                # Compute k1
                input_seq_current, update = self.explicit_update(input_seq_current)
                if self.mode == "integrator_training":
                    input_seq_current = self.integrator([update, input_seq_current])
                res.append(input_seq_current)
            res = tf.stack(res, axis=1)
            state_var_next = res[:,:,:,:,:2]
            velocity_next = res[:,:,:,:,2:]
            total_loss = (tf.keras.losses.MeanAbsoluteError(reduction='auto')(state_var_next,state_var_gt) + 
                          tf.keras.losses.MeanAbsoluteError(reduction='auto')(velocity_next,velocity_gt))/2
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
        #boundary_condition = input_seq_current[:, 0, 0, :]
        res = []
        for ts in range(self.n_time_step):
            input_seq_current, update = self.explicit_update(input_seq_current)
            if self.mode == "integrator_training":
                input_seq_current = self.integrator([update, input_seq_current])
            # Boundary condition reset
            #input_seq_current[:, :, 0, :] = boundary_condition[:, None, :]
            res.append(input_seq_current)
        res = tf.stack(res, axis=1)
        state_var_next = res[:,:,:,:,:2]
        velocity_next = res[:,:,:,:,2:]            
        total_loss  = (tf.keras.losses.MeanAbsoluteError(reduction='auto')(state_var_next,state_var_gt) + 
                       tf.keras.losses.MeanAbsoluteError(reduction='auto')(velocity_next,velocity_gt))/2

        self.total_loss_tracker.update_state(total_loss)

        return {
            "val_loss": self.total_loss_tracker.result(),
        }
    
    # Update scheme
    def explicit_update(self, input_seq_current):
        # Recommend refactor into directly using Butcher table
        if self.solver == "rk4":
            input_seq_current, update = self.rk4_update(input_seq_current)
        elif self.solver == 'heun':
            input_seq_current, update = self.heun_update(input_seq_current)
        elif self.solver == 'ode3':
            input_seq_current, update = self.ode3_update(input_seq_current)
        elif self.solver == "dormand_prince":
            input_seq_current, update = self.dormand_prince_update(input_seq_current)
        else:
            input_seq_current, update = self.euler_update(input_seq_current)

        return input_seq_current, update

    def dormand_prince_update(self, input_seq_current):
        '''
        Dormand-Prince method, the same method behind Matlab solve ode5. 5th order.
        '''
        input_seq_current = tf.clip_by_value(input_seq_current, 0, 1)
        k1 = self.differentiator(input_seq_current)
        k2 = self.differentiator(input_seq_current + 1/5 * self.step_size * k1)
        inp_k3 = input_seq_current + (3/40 * k1 + 9/40 * k2) * self.step_size
        k3 = self.differentiator(inp_k3)
        inp_k4 = input_seq_current + (44/45 * k1 - 56/15 * k2 + 32/9 * k3) * self.step_size
        k4 = self.differentiator(inp_k4)
        inp_k5 = input_seq_current + (19372/6561 * k1 - 25360/2187 * k2 + 64448/6561 * k3 - 212/729 * k4) * self.step_size
        k5 = self.differentiator(inp_k5)
        inp_k6 = input_seq_current + (9017/3168 * k1 - 355/33 * k2 + 46732/5247 * k3 + 49/176 * k4 - 5103/18656 * k5) * self.step_size
        k6 = self.differentiator(inp_k6)
        # Final
        update = 35/384 * k1 + 500/1113 * k3 + 125/192 * k4 - 2187/6784 * k5 + 11/84 * k6
        input_seq_current = input_seq_current + self.step_size * update
        return input_seq_current, update


    def ode3_update(self, input_seq_current):
        '''
        Bogacki–Shampine, the same method behind Matlab solver ode3. 3rd order.
        '''
        input_seq_current = tf.clip_by_value(input_seq_current, 0, 1)
        k1 = self.differentiator(input_seq_current)
        k2 = self.differentiator(input_seq_current + 1/2 * self.step_size * k1)
        k3 = self.differentiator(input_seq_current + 3/4 * self.step_size * k2)
        update = 2/9 * k1 + 1/3 * k2 + 4/9 * k3
        input_seq_current = input_seq_current + self.step_size * update
        return input_seq_current, update

    def rk4_update(self, input_seq_current):
        '''
        Original RK method. 4th order
        '''
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
        '''
        Heun's method. 2nd order
        '''
        input_seq_current = tf.clip_by_value(input_seq_current, 0, 1)
        # Compute k1
        k1 = self.differentiator(input_seq_current)
        # Compute k2
        inp_k2 = input_seq_current + self.step_size*k1 
        k2 = self.differentiator(inp_k2)
        # Final 
        update = 1/2*(k1 + k2)
        input_seq_current = input_seq_current + self.step_size*update 
        return input_seq_current, update
    
    # Euler update function
    def euler_update(self, input_seq_current):
        '''
        Euler's method. 1st order
        '''
        input_seq_current = tf.clip_by_value(input_seq_current, 0, 1)
        # Compute update
        update = self.differentiator(input_seq_current)
        input_seq_current = input_seq_current + self.step_size*update 
        return input_seq_current, update