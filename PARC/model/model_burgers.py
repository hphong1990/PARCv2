from tensorflow import keras
from tensorflow.keras import  layers, regularizers
from keras.layers import *
import tensorflow as tf
from parc import layer

from tensorflow.keras.layers import Concatenate, Input
from tensorflow.keras.models import Model
from parc.model.base_model import PARCv2

class PARCv2_burgers(PARCv2):
    def __init__(self, n_time_step, step_size, solver = "rk4", mode = "integrator_training", use_data_driven_int = True, *args, **kwargs):
        
        """
        Initilization of PARCv2 class:
        - n_time_step (float32): Define number of time step
        - step_size (float32): define the time step size (usually = 1/n_time_step, but depend on users)
        - solver (string): solver type, current available solver: Runge-kutta 4th order, Euler 2nd order, Euler 1st order
        - mode (string): training mode (integrator_training or differentiator_training). 
                For differentiator training: numerical integrator will be called
                For integrator training: hybrid integrator will be called, differentiator will be frozen
        - use_data_driven_int (bool): define if hybrid integrator or only numerical integrator will be used. Default: True

        """
        super(PARCv2_burgers, self).__init__(**kwargs)
        # Parse input
        self.n_time_step = n_time_step 
        self.step_size = step_size 
        self.solver = solver 
        self.use_data_driven_int = use_data_driven_int

        # Construct differentiator
        self.differentiator = self.build_differentiator()

        # Construct integrator
        self.integrator = self.build_integrator()

        # Create loss metric
        self.total_loss_tracker = keras.metrics.Mean(name="total_loss")

        # Define training mode
        if self.mode == "integrator_training":
            self.differentiator.trainable = False
        else:
            self.integrator.trainable = False


    def build_differentiator(self):
        """
        Differentiator definition customized for Burgers' problems: 
            - In this problem, state vars and velocity only include velocity
            - Architecture was adjusted to make it lighter and comparable with PhyCRNet
        """
        # Initialize neural networks
        # Create neural network for reaction term
        feature_extraction = layer.feature_extraction_burgers(input_shape = (64,64), n_channel = 3)
        
        # Create neural network for advection and diffusion terms
        advection = [layer.Advection() for _ in range(2)]
        diffusion = [layer.Diffusion() for _ in range(2)]

        # Create neural network for combining all terms and compute the time derivatives
        velocity_mapping_and_recon = layer.mapping_and_recon_cnn(input_shape = (64,64), n_base_features = 64, n_mask_channel=4, output_channel=2)

        # Main computation graph
        # Input 
        velocity_field = Input(shape=(64,64, 3), dtype = tf.float32)

        # Compute reaction term
        dynamic_feature = feature_extraction(velocity_field)

        # Compute the advection and diffusion for u (vel in x direction)
        advec_u = advection[0](velocity_field[:, :, :, 0:1], velocity_field[:, :, :, 0:2])
        diffusion_u = diffusion[0](velocity_field[:, :, :, 0:1])
        
        # Compute the advection and diffusion for v (vel in y direction)
        advec_v = advection[1](velocity_field[:, :, :, 1:2], velocity_field[:, :, :, 0:2])
        diffusion_v = diffusion[1](velocity_field[:, :, :, 1:2])    
        
        # Concatenate
        advec_diff_concat = Concatenate(axis=-1)([advec_u,advec_v,diffusion_u,diffusion_v])
        
        # Final mapping
        velocity_dot = velocity_mapping_and_recon([dynamic_feature, advec_diff_concat])
        
        # Create differentiator model
        differentiator = Model(velocity_field, velocity_dot, name = 'differentiator')
        return differentiator

    def build_integrator(self):
        # Initialize neural networks
        # Create neural network for integrator
        velocity_integrator = layer.integrator_cnn(input_shape = (64,64), n_base_features = 64, n_output=2)

        # Input layer
        velocity_prev = keras.layers.Input(shape = (64,64, 2), dtype = tf.float32)
        velocity_dot = keras.layers.Input(shape = (64,64, 2), dtype = tf.float32)

        # Output
        velocity_next = velocity_integrator([velocity_dot, velocity_prev])

        # Create integrator model
        integrator = keras.Model([velocity_dot, velocity_prev], [velocity_next], name = 'integrator')
        return integrator
    
    @property
    def metrics(self):
        return [
        self.total_loss_tracker,
        ]
    
    def call(self, input_tensor):
        # Convert input to tf_float32
        input_seq_current = tf.cast(input_tensor,dtype = tf.float32)

        # Initialize the output
        res = [] 
        # Add first step
        res.append(input_seq_current)
        
        if self.use_data_driven_int == True:    # If data-driven int is used
            for _ in range(self.n_time_step):
                # Solve using numerical integration
                velocity_next, update = self.explicit_update(input_seq_current)

                # Computing the high-order term using hybrid int and add to previous integration result
                velocity_next_hyper = self.integrator([update, velocity_next[:,:,:,:2]])

                # Combine output with constant for next step
                input_seq_current = Concatenate(axis = -1)([velocity_next_hyper, input_seq_current[:,:,:,2:]])

                # Add result to output list
                res.append(velocity_next_hyper)

        else:   # If data-driven int is not used
            for _ in range(self.n_time_step):  
                # Solve using numerical integration
                velocity_next, update = self.explicit_update(input_seq_current)

                # Get result and append to the final output
                input_seq_current = velocity_next[:,:,:,:2]
                res.append(input_seq_current)
        # Concatenate to make output array
        output = tf.concat(res,axis = -1)
        return output

    @tf.function
    def train_step(self, data):
        # Convert input tensor to float32 (for more efficient training)
        velocity_init = tf.cast(data[0], dtype = tf.float32)
        velocity_gt = tf.cast(data[1], dtype = tf.float32)

        # Initialize the training
        input_seq_current = velocity_init

        # Start of the gradient recording
        with tf.GradientTape() as tape:
            output_snap = []
            if self.mode == "integrator_training":  # If training the integrator
                for _ in range(self.n_time_step):
                    # Solve using numerical integration

                    velocity_next, update = self.explicit_update(input_seq_current)

                    # Compute the high order term using hybrid int and add to previous output
                    velocity_next_hyper = self.integrator([update, velocity_next[:,:,:,:2]])

                    # Combine result with constant to make input for next step
                    input_seq_current = Concatenate(axis = -1)([velocity_next_hyper, velocity_init[:,:,:,2:]])

                    # Clip value to avoid numerical instability when training with long sequence
                    input_seq_current = tf.clip_by_value(input_seq_current,0,1)

                    # Append the result to the output list
                    output_snap.append(input_seq_current[:,:,:,:2])
            else:   # If training the differentiator
                for _ in range(self.n_time_step):
                    # Solve using numerical integration
                    velocity_next, update = self.explicit_update(input_seq_current)

                    # Make input for the next step
                    input_seq_current = velocity_next

                    # Clip value to avoid numerical instability
                    input_seq_current = tf.clip_by_value(input_seq_current,0,1)

                    # Append the result to the output list
                    output_snap.append(input_seq_current[:,:,:,:2])
            
            # Make prediction array
            velocity_pred = Concatenate(axis = -1)(output_snap)

            # Compute the loss (using MAE loss)
            total_loss  = tf.keras.losses.MeanAbsoluteError(reduction = 'sum')(velocity_pred,velocity_gt)
        # Compute the gradient                           
        grads = tape.gradient(total_loss, self.trainable_weights)

        # Update weights
        self.optimizer.apply_gradients(zip(grads, self.trainable_weights))

        # Update loss value
        self.total_loss_tracker.update_state(total_loss)

        return {
            "total_loss": self.total_loss_tracker.result(),
        }
    
    # # Update scheme
    # def explicit_update(self, input_seq_current):
    #     input_seq_current = tf.clip_by_value(input_seq_current, 0, 1)

    #     if self.solver == "rk4":
    #         input_seq_current, update = self.rk4_update(input_seq_current)
    #     elif self.solver == 'heun':
    #         input_seq_current, update = self.heun_update(input_seq_current)
    #     else:
    #         input_seq_current, update = self.euler_update(input_seq_current)

    #     return input_seq_current, update

    # def rk4_update(self, input_seq_current):

    #     # Compute k1
    #     k1 = self.differentiator(input_seq_current)

    #     # Compute k2
    #     inp_k2 = input_seq_current[:,:,:,:2] + self.step_size*1/2*k1 
    #     inp_k2 = Concatenate(axis = -1)([inp_k2,input_seq_current[:,:,:,2:]])

    #     k2 = self.differentiator(inp_k2)

    #     # Compute k3
    #     inp_k3 = input_seq_current[:,:,:,:2] + self.step_size*1/2*k2
    #     inp_k3 = Concatenate(axis = -1)([inp_k3,input_seq_current[:,:,:,2:]])
    #     k3 = self.differentiator(inp_k3)

    #     # Compute k4
    #     inp_k4 = input_seq_current[:,:,:,:2] + self.step_size*k3
    #     inp_k4 = Concatenate(axis = -1)([inp_k4,input_seq_current[:,:,:,2:]])

    #     k4 = self.differentiator(inp_k4)

    #     # Final
    #     update = 1/6*(k1 + 2*k2 + 2*k3 + k4)
    #     final_state = input_seq_current[:,:,:,:2] + self.step_size*update 
    #     input_seq_current = Concatenate(axis = -1)([final_state,input_seq_current[:,:,:,2:]])
    #     return input_seq_current, update
    
    # # Euler update function
    # def heun_update(self, input_seq_current):
    #     # Compute update
    #     k1 = self.differentiator(input_seq_current)

    #     # Compute k2
    #     inp_k2 = input_seq_current[:,:,:,:2] + self.step_size*k1 
    #     inp_k2 = Concatenate(axis = -1)([inp_k2,input_seq_current[:,:,:,2:]])

    #     k2 = self.differentiator(inp_k2)
        
    #     update = 1/2*(k1 + k2)

    #     final_states = input_seq_current[:,:,:,:2] + self.step_size*update 
    #     input_seq_current = Concatenate(axis = -1)([final_states,input_seq_current[:,:,:,2:]])

    #     return input_seq_current, update
    
    # # Euler update function
    # def euler_update(self, input_seq_current):
    #     # Compute update
    #     update = self.differentiator(input_seq_current)
    #     input_seq_current = input_seq_current + self.step_size*update 

    #     return input_seq_current, update