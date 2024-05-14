from tensorflow import keras
from tensorflow.keras import  layers, regularizers
from keras.layers import *
import tensorflow as tf
from parc import layer

from tensorflow.keras.layers import Concatenate, Input
from tensorflow.keras.models import Model
from parc.model.base_model import PARCv2

class PARCv2_ns(PARCv2):
    def __init__(self, n_time_step, step_size, solver = "rk4", mode = "integrator_training", use_data_driven_int = True, differentiator_backbone = 'em', **kwargs):
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
        super(PARCv2_ns, self).__init__(**kwargs)

        # Parse input
        self.n_time_step = n_time_step
        self.step_size = step_size
        self.solver = solver
        self.mode = mode
        self.use_data_driven_int = use_data_driven_int

        # Construct differentiator
        self.differentiator = self.build_differentiator()

        # Construct poisson block

        self.poisson = self.poisson_block()

        # Construct integrator
        self.integrator = self.build_integrator()

        # Construct integrator
        self.total_loss_tracker = keras.metrics.Mean(name="total_loss")
        if self.mode == "integrator_training":
            self.differentiator.trainable = False
            self.poisson.trainable = False
        else:
            self.integrator.trainable = False
       
    def build_differentiator(self, input_shape = (128,256), n_out_features = 64, n_base_features = 64):
        """
        Differentiator definition customized for Burgers' problems: 
            - In this problem, state vars and velocity only include velocity
            - Architecture was adjusted to make it lighter
        """

        # Initialize neural networks
        # Create neural network for reaction term
        feature_extraction = layer.feature_extraction_unet(input_shape = input_shape, n_out_features = n_out_features, n_base_features = n_base_features, n_channel = 3)
        
        # Create neural network for advection and diffusion terms

        advection = [layer.Advection() for _ in range(2)]
        diffusion = [layer.Diffusion() for _ in range(2)]

        # Create neural network for combining all terms and compute the time derivatives
        velocity_mapping_and_recon = layer.mapping_and_recon_cnn(input_shape = input_shape, n_base_features = n_base_features, n_mask_channel=4, output_channel=2)

        # Main computation graph
        # Input layer
        velocity_field = Input(shape=(input_shape[0],input_shape[1], 3), dtype = tf.float32)

        # Compute reaction term
        dynamic_feature = feature_extraction(velocity_field)

        # Compute advection and diffusion terms for u
        advec_u = advection[0](velocity_field[:, :, :, 0:1], velocity_field[:, :, :, :2])
        diffusion_u = diffusion[0](velocity_field[:, :, :, 0:1])
        
        # Compute advection and diffusion terms for v
        advec_v = advection[1](velocity_field[:, :, :, 1:2], velocity_field[:, :, :, :2])
        diffusion_v = diffusion[1](velocity_field[:, :, :, 1:2])    
        
        # Concatenate advection and diffusion terms
        advec_diff_concat = Concatenate(axis=-1)([advec_u,advec_v,diffusion_u,diffusion_v])
        
        # Combine and compute the time derivatives
        velocity_dot = velocity_mapping_and_recon([dynamic_feature, advec_diff_concat])
        
        # Create the differentiator model
        differentiator = Model(velocity_field, velocity_dot, name = 'differentiator')
        return differentiator

    def poisson_block(self, input_shape = (128,256),n_base_features = 64):
        """
        One extra neural network to compute the pressure terms
        """
        # Input layer
        inputs = keras.Input(shape = (input_shape[0], input_shape[1], 3), dtype = tf.float32)

        # Poisson layers
        poisson = layer.Poisson()([inputs[:,:,:,0:1],inputs[:,:,:,1:2]])
        
        concat = Concatenate(axis = -1)([inputs,poisson[0], poisson[1],poisson[2]])

        # Combine all input and compute features
        conv = layer.conv_block_down(concat,
                            feat_dim = n_base_features,
                                reps = 1,
                                kernel_size = 3,
                                mode = 'normal')
        
        # Resnet block to compute the pressure
        conv_res = layer.resnet_block(conv, n_base_features, kernel_size = 3, reps = 2, pooling = False)
        conv_out = Conv2D(1,1, padding='same')(conv_res)

        # Create poisson model
        poisson = keras.Model([inputs], conv_out,  name = 'possion')
        return poisson
    
    def build_integrator(self, input_shape = (128, 256)):
        """
        Build the integrator for the Navier-Stokes problem

        """

        # Input layer
        state_var_prev = keras.layers.Input(shape = (input_shape[0], input_shape[1], 2), dtype = tf.float32)
        state_var_dot = keras.layers.Input(shape = (input_shape[0], input_shape[1], 2), dtype = tf.float32)

        # Convolutional layer to compute the high-order terms
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
        
        # Add the high-order term to the input to compute the final output
        state_var_next = Add()([state_var_prev, conv_out])

        # Create the integrator network
        integrator = keras.Model([state_var_dot, state_var_prev], [state_var_next], name = 'integrator')
        return integrator
    
    @property
    def metrics(self):
        return [
        self.total_loss_tracker,
        ]
    
    def call(self, input_data):
        """
        Inference function, use to make prediction after training

        """
        # Convert input to tf_float32
        input_seq_current = tf.cast(input_data,dtype = tf.float32)

        # Initialize the output
        res = []
        res.append(input_seq_current)

        if self.use_data_driven_int == True:    # If data-driven int is used
            for ts in range(self.n_time_step):
                # Compute the pressure
                pressure = self.poisson(input_seq_current)

                # Combine pressure with previous velocity variables
                input_seq_current = Concatenate(axis = -1)([input_seq_current[:,:,:,:2], pressure])

                # Solve using numerical integration
                input_seq_current, update = self.explicit_update(input_seq_current)

                # Compute the high-order term and add to the previous computation
                velocity_next = self.integrator([update, input_seq_current[:,:,:,:2]])

                # Combine prediction value with constant to serve as input for the next step
                input_seq_current = Concatenate(axis = -1)([velocity_next, input_seq_current[:,:,:,2:3]])

                # Clip data to have value from 0 -1 
                input_seq_current = tf.clip_by_value(input_seq_current,0,1)

                # Add prediction to output list
                res.append(input_seq_current[:,:,:,:3])
            # Make prediction array
            output = Concatenate(axis = -1)(res)
                
        else:
            for ts in range(self.n_time_step):
                # Compute the pressure
                pressure = self.poisson(input_seq_current)

                # Combine pressure with previous velocity variables
                input_seq_current = Concatenate(axis = -1)([input_seq_current[:,:,:,:2], pressure])

                # Solve using numerical integration
                input_seq_current, update = self.explicit_update(input_seq_current)

                # Clip data to have value from 0 -1 
                input_seq_current = tf.clip_by_value(input_seq_current,0,1)
                
                # Add prediction to output list
                res.append(input_seq_current[:,:,:,:3])

            # Make prediction array
            output = Concatenate(axis = -1)(res)      
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
            if self.mode == "integrator_training":  # If training the integrator
                output_snap = []
                for _ in range(self.n_time_step):
                    # Compute pressure
                    pressure = self.poisson(input_seq_current)
                    input_seq_current = Concatenate(axis = -1)([input_seq_current[:,:,:,:2], pressure])

                    # Solve using numerical integration
                    input_seq_current, update = self.explicit_update(input_seq_current)

                    # Compute the high order term using hybrid int and add to previous output
                    velocity_next = self.integrator([update, input_seq_current[:,:,:,:2]])

                    # Combine result with constant to make input for next step
                    input_seq_current = Concatenate(axis = -1)([velocity_next, input_seq_current[:,:,:,2:3]])

                    # Clip value
                    input_seq_current = tf.clip_by_value(input_seq_current,0,1)

                    # Add result to prediction list
                    output_snap.append(input_seq_current[:,:,:,:3])

                # Make prediction array
                output = Concatenate(axis = -1)(output_snap)
                
            else:
                output_snap = []
                for _ in range(self.n_time_step):
                    # Compute pressure
                    pressure = self.poisson(input_seq_current)
                    input_seq_current = Concatenate(axis = -1)([input_seq_current[:,:,:,:2], pressure])
                    # Solve using numerical integration
                    input_seq_current, update = self.explicit_update(input_seq_current)

                    # Clip value
                    input_seq_current = tf.clip_by_value(input_seq_current,0,1)

                    # Add to the prediction list
                    output_snap.append(input_seq_current[:,:,:,:3])

                # Make prediction array
                output = Concatenate(axis = -1)(output_snap)

            # Compute loss                
            total_loss = tf.keras.losses.MeanAbsoluteError(reduction = 'sum')(output,velocity_gt) 
        
        # Compute gradient                   
        grads = tape.gradient(total_loss, self.trainable_weights)

        # Update weights
        self.optimizer.apply_gradients(zip(grads, self.trainable_weights))

        # Update loss
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