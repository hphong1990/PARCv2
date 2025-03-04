from tensorflow import keras
from tensorflow.keras import  layers, regularizers
from keras.layers import *
import tensorflow as tf
from parc import layer

from tensorflow.keras.layers import Concatenate, Input
from tensorflow.keras.models import Model
from parc.model.base_model import PARCv2

class PARCv2_EM(PARCv2):
    def __init__(self, n_state_var, n_time_step, step_size, m_input_shape = (128,192), solver = "rk4", mode = "integrator_training", use_data_driven_int = True, *args,  **kwargs):
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
        super(PARCv2_EM, self).__init__(**kwargs)
        # Parse input
        self.n_state_var = n_state_var
        self.n_time_step = n_time_step
        self.step_size = step_size
        self.solver = solver
        self.mode = mode
        self.use_data_driven_int = use_data_driven_int
        self.input_shape = m_input_shape
        # Construct differentiator
        self.differentiator = self.build_differentiator(input_shape = self.input_shape, n_state_var=self.n_state_var)

        # Construct integrator
        self.integrator = self.build_integrator(input_shape = self.input_shape, n_state_var=self.n_state_var)

        # Create loss metric
        self.total_loss_tracker = keras.metrics.Mean(name="total_loss")

        # Define training mode
        if self.mode == "integrator_training":
            self.differentiator.trainable = False
        else:
            self.integrator.trainable = False
    
    def build_differentiator(self, n_state_var=3, m_input_shape = (128, 192)):
        """
        Differentiator for EM problems: 
            - State vars (3) include temperature, pressure, microstructure evolution
            - Velocity vars(2) include velocity in 2 directions (x,y)
            - There is no constant field
        """
        # Initialize neural networks
        # Create neural network for reaction term
        feature_extraction = layer.feature_extraction_unet(input_shape = m_input_shape, n_channel=n_state_var+2)
        
        
        advection = [layer.Advection() for _ in range(n_state_var+2)]
        diffusion = layer.Diffusion()

        # Create neural network for combining all terms and compute the time derivatives
        # For state variables
        mapping_and_recon = []
        mapping_and_recon.append(layer.mapping_and_recon_cnn(input_shape = m_input_shape, n_mask_channel=2, output_channel=1))
        mapping_and_recon.append(layer.mapping_and_recon_cnn(input_shape = m_input_shape, n_mask_channel=1, output_channel=1))
        mapping_and_recon.append(layer.mapping_and_recon_cnn(input_shape = m_input_shape, n_mask_channel=1, output_channel=1))
        
        # For velocity variables
        velocity_mapping_and_recon = layer.mapping_and_recon_cnn(input_shape = m_input_shape, n_mask_channel=2, output_channel=2)

        # Main computation graph
        # Input
        input_tensor = Input(shape=(m_input_shape[0] , m_input_shape[1], n_state_var+2), dtype = tf.float32)
        init_state_var = input_tensor[:,:,:,:n_state_var]
        velocity_field = input_tensor[:,:,:,n_state_var:]

        # Compute reaction term
        dynamic_feature = feature_extraction(input_tensor)

        # Compute the advection and diffusion for temperature
        advec_temp = advection[0](init_state_var[:, :, :, 0:1], velocity_field)
        diffusion_temp = diffusion(init_state_var[:, :, :, 0:1])
        temp_concat = Concatenate(axis=-1)([advec_temp,diffusion_temp])
        temp_dot = mapping_and_recon[0]([dynamic_feature, temp_concat])
        
        # Compute the advection and diffusion for pressure
        advec_press = advection[1](init_state_var[:, :, :, 1:2], velocity_field)
        press_dot = mapping_and_recon[1]([dynamic_feature, advec_press])
        
        # Compute the advection and diffusion for microstructure
        advec_micro = advection[2](init_state_var[:, :, :, 2:3], velocity_field)
        micro_dot = mapping_and_recon[2]([dynamic_feature, advec_micro])
        
        # Compute the advection and diffusion for microstructure for velocity
        advec_vel = []
        for i in range(2):
            advec_i = advection[i+3](velocity_field[:, :, :, i:i+1], velocity_field)
            advec_vel.append(advec_i)

        # Concatenate
        advec_vel_concat = Concatenate(axis=-1)(advec_vel)

        # Combine and compute time derivatives
        velocity_dot = velocity_mapping_and_recon([dynamic_feature, advec_vel_concat])
        output_tensor = Concatenate(axis=-1)([temp_dot,press_dot,micro_dot, velocity_dot])
        
        # Create differentiator model
        differentiator = Model(input_tensor, output_tensor)
        return differentiator

    def build_integrator(self, m_input_shape = (128, 192), n_state_var = 3):
        """
        Integrator definition customized for EM problems

        """
        
        # Initialize neural networks
        # Create neural network for state variable integration
        state_integrators = []
        for _ in range(n_state_var):
            state_integrators.append(layer.integrator_cnn(input_shape = m_input_shape))

        # Create neural network for velocity variable integration
        velocity_integrator = layer.integrator_cnn(input_shape = m_input_shape, n_output=2)

        # Input layers

        # Previous steps values input
        state_var_prev = keras.layers.Input(shape = (m_input_shape[0], m_input_shape[1], n_state_var), dtype = tf.float32)
        velocity_prev = keras.layers.Input(shape = (m_input_shape[0], m_input_shape[1],2), dtype = tf.float32)
        # Time derivatives input
        state_var_dot = keras.layers.Input(shape = (m_input_shape[0], m_input_shape[1],n_state_var), dtype = tf.float32)
        velocity_dot = keras.layers.Input(shape = (m_input_shape[0], m_input_shape[1],2), dtype = tf.float32)
        
        # Computing the high order term and add to the previous step value
        state_var_next = []

        # Compute next step value for state variables
        for i in range(n_state_var): 
            state_var_next.append(state_integrators[i]([state_var_dot[:,:,:,i:i+1], state_var_prev[:,:,:,i:i+1]]))
        state_var_next = keras.layers.concatenate(state_var_next, axis=-1)

        # Compute next step value for velocity variables
        velocity_next = velocity_integrator([velocity_dot, velocity_prev])

        # Create neural network model
        integrator = keras.Model([state_var_dot, velocity_dot, state_var_prev, velocity_prev], [state_var_next, velocity_next])
        return integrator

    @property
    def metrics(self):
        return [
        self.total_loss_tracker,
        ]
    
    def call(self, input):
        """
        Inference function, use to make prediction after training
        """

        # Convert input to tf_float32
        state_var_init = tf.cast(input[0],dtype = tf.float32)
        velocity_init = tf.cast(input[1], dtype = tf.float32)
        input_seq = Concatenate(axis = -1)([state_var_init, velocity_init])

        # Initialize the output
        input_seq_current = input_seq
        res = []

        for _ in range(self.n_time_step):    
            # Compute next step value using numerical integration
            input_seq_current, update = self.explicit_update(input_seq_current)
            if self.use_data_driven_int == True:    # If use data-driven integration
                # Compute high-order term and add to previous prediction
                state_var_next, velocity_next = self.integrator([update[:,:,:,:3],update[:,:,:,3:],input_seq_current[:,:,:,:3], input_seq_current[:,:,:,3:]])
                input_seq_current = Concatenate()([state_var_next, velocity_next])
            # Add prediction to the output list                        
            res.append(input_seq_current)
        return res

    @tf.function
    def train_step(self, data):
        # Convert input tensor to float32 (for more efficient training)
        state_var_init = tf.cast(data[0][0],dtype = tf.float32)
        velocity_init = tf.cast(data[0][1], dtype = tf.float32)
        input_seq = Concatenate(axis = -1)([state_var_init, velocity_init])

        state_var_gt = tf.cast(data[1][0], dtype = tf.float32)
        velocity_gt = tf.cast(data[1][1], dtype = tf.float32)

        # Initialize the training
        input_seq_current = input_seq

        # Start of the gradient recording
        with tf.GradientTape() as tape:
            state_whole = []
            vel_whole = []
            if self.mode == "integrator_training": # If training the integrator
                for ts in range(self.n_time_step):
                    # Solve using numerical integration
                    input_seq_current, update = self.explicit_update(input_seq_current)
                    
                    # Compute the high order term using hybrid int and add to previous output
                    state_var_next, velocity_next = self.integrator([update[:,:,:,:3],update[:,:,:,3:],input_seq_current[:,:,:,:3], input_seq_current[:,:,:,3:]])
                    
                    # Combine result for state variable and velocity variable to make the input for next step
                    input_seq_current = Concatenate()([state_var_next, velocity_next])

                    # Add prediction result to output list
                    state_whole.append(state_var_next)
                    vel_whole.append(velocity_next)
            else: # If training the differentiator
                for ts in range(self.n_time_step):
                    # Solve using numerical integration
                    input_seq_current, update = self.explicit_update(input_seq_current)

                    # Add prediction result to output list
                    state_whole.append( input_seq_current[:,:,:,:3])
                    vel_whole.append(input_seq_current[:,:,:,3:])
            
            # Make prediction array
            state_pred = Concatenate(axis = -1)(state_whole)
            vel_pred = Concatenate(axis = -1)(vel_whole)

            # Compute loss (using MAE)        
            total_loss  = (tf.keras.losses.MeanAbsoluteError(reduction = 'sum')(state_pred,state_var_gt) + 
                            tf.keras.losses.MeanAbsoluteError(reduction = 'sum')(vel_pred,velocity_gt))/2
        # Compute gradient
        grads = tape.gradient(total_loss, self.trainable_weights)

        # Update parameters
        self.optimizer.apply_gradients(zip(grads, self.trainable_weights))

        # Update loss value
        self.total_loss_tracker.update_state(total_loss)

        return {
            "total_loss": self.total_loss_tracker.result(),
        }
    
    # # Update scheme
    # def explicit_update(self, input_seq_current):
    #     if self.solver == "rk4":
    #         input_seq_current, update = self.rk4_update(input_seq_current)
    #     elif self.solver == 'heun':
    #         input_seq_current, update = self.heun_update(input_seq_current)
    #     else:
    #         input_seq_current, update = self.euler_update(input_seq_current)

    #     return input_seq_current, update

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