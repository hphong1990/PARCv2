from tensorflow import keras
from tensorflow.keras import  layers, regularizers
from keras.layers import *
import tensorflow as tf
from parc import layer

from tensorflow.keras.layers import Concatenate, Input
from tensorflow.keras.models import Model

def differentiator_ns():
    # Model initiation
    feature_extraction = layer.feature_extraction_unet(input_shape = (128,256), n_out_features = 64, n_base_features = 64, n_channel = 3)
    
    advection = [layer.Advection() for _ in range(2)]
    diffusion = [layer.Diffusion() for _ in range(2)]
    velocity_mapping_and_recon = layer.mapping_and_recon_cnn(input_shape = (128,256), n_base_features = 64, n_mask_channel=4, output_channel=2)

    # Main computation graph
    velocity_field = Input(shape=(128,256, 3), dtype = tf.float32)

    # Reaction term
    dynamic_feature = feature_extraction(velocity_field)

    # u
    advec_u = advection[0](velocity_field[:, :, :, 0:1], velocity_field[:, :, :, :2])
    diffusion_u = diffusion[0](velocity_field[:, :, :, 0:1])
    
    # v
    advec_v = advection[1](velocity_field[:, :, :, 1:2], velocity_field[:, :, :, :2])
    diffusion_v = diffusion[1](velocity_field[:, :, :, 1:2])    
    
    # Concatenate
    advec_diff_concat = Concatenate(axis=-1)([advec_u,advec_v,diffusion_u,diffusion_v])
    
    # Final mapping
    velocity_dot = velocity_mapping_and_recon([dynamic_feature, advec_diff_concat])
    
    differentiator = Model(velocity_field, velocity_dot, name = 'differentiator')
    return differentiator


def poisson_block(input_shape = (128,256),n_base_features = 64):
    inputs = keras.Input(shape = (input_shape[0], input_shape[1], 3), dtype = tf.float32)
    poisson = layer.Poisson()([inputs[:,:,:,0:1],inputs[:,:,:,1:2]])
                               
    concat = Concatenate(axis = -1)([inputs,poisson[0], poisson[1],poisson[2]])
    conv = layer.conv_block_down(concat,
                           feat_dim = n_base_features,
                            reps = 1,
                            kernel_size = 3,
                            mode = 'normal')
    conv_res = layer.resnet_block(conv, n_base_features, kernel_size = 3, reps = 2, pooling = False)
    conv_out = Conv2D(1,1, padding='same')(conv_res)
    poisson = keras.Model([inputs], conv_out,  name = 'possion')
    return poisson
 
def integrator():
    state_var_prev = keras.layers.Input(shape = (128, 256, 2), dtype = tf.float32)
    state_var_dot = keras.layers.Input(shape = (128, 256,2), dtype = tf.float32)

    conv = layer.conv_block_down(state_var_dot,
                           feat_dim = 64,
                            reps = 1,
                            kernel_size = 5,
                            mode = 'down')
    conv2 = layer.conv_block_down(conv,
                           feat_dim = 128,
                            reps = 1,
                            kernel_size = 5,
                            mode = 'normal')
    conv3 = layer.conv_block_up_wo_concat(conv2,
                           feat_dim = 64,
                            reps = 1,
                            kernel_size = 5,
                            mode = 'up')
    conv_out = Conv2D(2,3, padding='same')(conv3)
    state_var_next = Add()([state_var_prev, conv_out])
    integrator = keras.Model([state_var_dot, state_var_prev], [state_var_next], name = 'integrator')
    return integrator

@keras.saving.register_keras_serializable()
class PARCv2_ns(keras.Model):
    def __init__(self, n_time_step, step_size, solver = "rk4", mode = "integrator_training", use_data_driven_int = True, differentiator_backbone = 'em', **kwargs):
        super(PARCv2_ns, self).__init__(**kwargs)
        self.n_time_step = n_time_step
        self.step_size = step_size
        self.solver = solver
        self.mode = mode
        self.use_data_driven_int = use_data_driven_int
        self.differentiator = differentiator_ns()
        self.poisson = poisson_block()
        self.integrator = integrator()
        self.total_loss_tracker = keras.metrics.Mean(name="total_loss")
        if self.mode == "integrator_training":
            self.differentiator.trainable = False
            self.poisson.trainable = False
        else:
            self.integrator.trainable = False

        self.input_layer = keras.layers.Input((128, 256, 3))
        self.out = self.call(self.input_layer)
        
        super(PARCv2_ns, self).__init__(
            inputs= self.input_layer, outputs=self.out, **kwargs
        )
    
    def build(self):
        self._is_graph_network = True
        self._init_graph_network(
            inputs=[self.input_layer], outputs=self.out
        )

    @property
    def metrics(self):
        return [
        self.total_loss_tracker,
        ]
    
    def call(self, input_data):
        input_seq_current = tf.cast(input_data,dtype = tf.float32)
        res = []
        res.append(input_seq_current)
        if self.use_data_driven_int == True:
            for ts in range(self.n_time_step):
                # Compute k1
                pressure = self.poisson(input_seq_current)
                input_seq_current = Concatenate(axis = -1)([input_seq_current[:,:,:,:2], pressure])
                input_seq_current, update = self.explicit_update(input_seq_current)
                velocity_next = self.integrator([update, input_seq_current[:,:,:,:2]])
                input_seq_current = Concatenate(axis = -1)([velocity_next, input_seq_current[:,:,:,2:3]])
                input_seq_current = tf.clip_by_value(input_seq_current,0,1)
                res.append(input_seq_current[:,:,:,:3])
            output = Concatenate(axis = -1)(res)
                
        else:
            for ts in range(self.n_time_step):
                pressure = self.poisson(input_seq_current)
                input_seq_current = Concatenate(axis = -1)([input_seq_current[:,:,:,:2], pressure])
                input_seq_current, update = self.explicit_update(input_seq_current)
                input_seq_current = tf.clip_by_value(input_seq_current,0,1)
                res.append(input_seq_current[:,:,:,:3])
            output = Concatenate(axis = -1)(res)      
        return output

    @tf.function
    def train_step(self, data):
        velocity_init = tf.cast(data[0], dtype = tf.float32)
        velocity_gt = tf.cast(data[1], dtype = tf.float32)

        input_seq_current = velocity_init
        with tf.GradientTape() as tape:
            if self.mode == "integrator_training":
                output_snap = []

                for ts in range(self.n_time_step):
                    pressure = self.poisson(input_seq_current)
                    input_seq_current = Concatenate(axis = -1)([input_seq_current[:,:,:,:2], pressure])
                    input_seq_current, update = self.explicit_update(input_seq_current)
                    velocity_next = self.integrator([update, input_seq_current[:,:,:,:2]])
                    input_seq_current = Concatenate(axis = -1)([velocity_next, input_seq_current[:,:,:,2:3]])
                    input_seq_current = tf.clip_by_value(input_seq_current,0,1)
                    output_snap.append(input_seq_current[:,:,:,:3])
                output = Concatenate(axis = -1)(output_snap)
                
            else:
                output_snap = []
                for ts in range(self.n_time_step):
                    pressure = self.poisson(input_seq_current)
                    input_seq_current = Concatenate(axis = -1)([input_seq_current[:,:,:,:2], pressure])
                    input_seq_current, update = self.explicit_update(input_seq_current)
                    input_seq_current = tf.clip_by_value(input_seq_current,0,1)
                    output_snap.append(input_seq_current[:,:,:,:3])
                output = Concatenate(axis = -1)(output_snap)
                
            total_loss = tf.keras.losses.MeanAbsoluteError(reduction = 'sum')(output,velocity_gt) 
                           
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
        inp_k2 = input_seq_current[:,:,:,1:] + self.step_size*1/2*k1
        inp_k2 = Concatenate(axis = -1)([input_seq_current[:,:,:,0:1],inp_k2])

        k2 = self.differentiator(inp_k2)

        # Compute k3
        inp_k3 = input_seq_current[:,:,:,1:] + self.step_size*1/2*k2
        inp_k3 = Concatenate(axis = -1)([input_seq_current[:,:,:,0:1],inp_k3])
        k3 = self.differentiator(inp_k3)

        # Compute k4
        inp_k4 = input_seq_current[:,:,:,1:] + self.step_size*k3
        inp_k4 = Concatenate(axis = -1)([input_seq_current[:,:,:,0:1],inp_k4])

        k4 = self.differentiator(inp_k4)

        # Final
        update = 1/6*(k1 + 2*k2 + 2*k3 + k4)
        final_state = input_seq_current[:,:,:,1:] + self.step_size*update 
        input_seq_current = Concatenate(axis = -1)([input_seq_current[:,:,:,0:1],final_state])
        return input_seq_current, update
    
    # Euler update function
    def heun_update(self, input_seq_current):
        # Compute update
        k1 = self.differentiator(input_seq_current)

        # Compute k2       
        inp_k2 = input_seq_current[:,:,:,0:2] + self.step_size*k1
        inp_k2 = Concatenate(axis = -1)([inp_k2,input_seq_current[:,:,:,2:]])

        k2 = self.differentiator(inp_k2)
        
        update = 1/2*(k1 + k2)
        
        final_state = input_seq_current[:,:,:,0:2] + self.step_size*update 
        input_seq_current = Concatenate(axis = -1)([final_state,input_seq_current[:,:,:,2:]])
        return input_seq_current, update
    
#     # Euler update function
#     def euler_update(self, input_seq_current):
#         # Compute update
#         update = self.differentiator(input_seq_current)
#         input_seq_current = input_seq_current + self.step_size*update 

#         return input_seq_current, update