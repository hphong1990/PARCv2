from tensorflow import keras
from tensorflow.keras import  layers, regularizers
from keras.layers import *
import tensorflow as tf
from PARC import layer

from tensorflow.keras.layers import Concatenate, Input
from tensorflow.keras.models import Model
from abc import abstractmethod

"""
This is the base PARCv2 class for model definition. 

This base class include the explicit update scheme that is used for integration

Users will need to subclassing this class and write their own differentiator and
integrator implementation. Also, 'train_step' and 'validation_step' functions will
also needed to be defined by the user

"""
class PARCv2(keras.Model):
    def __init__(self, **kwargs):
        super(PARCv2, self).__init__(**kwargs)

    @property
    def metrics(self):
        return [
        self.total_loss_tracker,
        ]

    # Update scheme
    def explicit_update(self, input_seq_current):
        # Clipping input tensor to have the value from 0 to 1
        input_seq_current = tf.clip_by_value(input_seq_current, 0, 1)

        # Select different solver
        if self.solver == "rk4": # Runge-kutta 4th order 
            input_seq_current, update = self.rk4_update(input_seq_current)
        elif self.solver == 'heun': # Euler 2nd order
            input_seq_current, update = self.heun_update(input_seq_current)
        else: # Default: Euler 1st order
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
        inp_k2 = input_seq_current[:,:,:,:2] + self.step_size*k1 
        inp_k2 = Concatenate(axis = -1)([inp_k2,input_seq_current[:,:,:,2:]])

        k2 = self.differentiator(inp_k2)
        
        # Final
        update = 1/2*(k1 + k2)

        final_states = input_seq_current[:,:,:,:2] + self.step_size*update 
        input_seq_current = Concatenate(axis = -1)([final_states,input_seq_current[:,:,:,2:]])

        return input_seq_current, update
    
    # Euler update function
    def euler_update(self, input_seq_current):
        # Compute update
        update = self.differentiator(input_seq_current)
        input_seq_current = input_seq_current + self.step_size*update 

        return input_seq_current, update