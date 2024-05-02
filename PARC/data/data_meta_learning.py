import time
import os
import numpy as np
import skimage
from skimage.measure import block_reduce

  
def clip_raw_data(sequence_length=2, n_state_var=3, purpose = "diff_training", folder_path='...', image_size = (128, 208)):
    
    # idx_range: is used for splitting data into training and testing data, we will do this automatically in the future
    # sequence_length: the amount of time steps contained for one sequence
    # n_state_var: is the number of state variables our model considers - PARCv2 considers: Temperature, Pressure, and Microstructure
    # purpose: diff_training will chop data with a sliding window of size sequence length from any starting time step; else int_training it will takes from time zero to sequence length 
    # folder_path: folder containing the data for the simulation
    # image_size: is the x and y dimension of the numpy data.. in tensorflow it is H and W
    
    
    # init list of training samples
    # state is for state variables
    # vel is for velocity
    state_seq_whole = []
    vel_seq_whole = []

    # Loop through each file in the folder
    for filename in os.listdir(folder_path):
        if filename.endswith(".npy"):
            file_path = os.path.join(folder_path, filename)
            print(f"Processing {filename}")
            raw_data = np.float32(np.load(file_path))
            data_shape = raw_data.shape
            
            # ensures the data will work for the sequence length
            if data_shape[2] > sequence_length:
                
                # create batch dimensions for training for TensorFlow
                raw_data = np.expand_dims(raw_data, axis=0)
                
                # reduce the spatial size of the height and width
                raw_data = skimage.measure.block_reduce(raw_data[:,:,:,:], (1,4,4,1),np.max)
                

                data_shape = raw_data.shape
                
                # calculate the number of time steps in the original data
                num_time_steps = data_shape[-1] // (n_state_var + 2)
                
                # j_range determines how the data should slide for time sequence
                # this indicates the maximum starting index you observe from 
                if purpose == "diff_training":
                    j_range = num_time_steps - sequence_length
                else:
                    # start from beginning to sequence length
                    j_range = 1
                    
                ### changing 128 x 208
                
                ### TO DO:  Diagram for explaining process 
                state_seq_case = [np.concatenate([raw_data[:, :image_size[0], :image_size[1], (j + k) * (n_state_var + 2):\
                                                        (j + k) * (n_state_var + 2) + n_state_var] \
                                                        for k in range(sequence_length)], axis=-1) \
                                                        for j in range  (j_range)] 

                vel_seq_case = [np.concatenate([raw_data[:, :image_size[0], :image_size[1], (j + k) * (n_state_var + 2) +  n_state_var :\
                                                        (j + k) * (n_state_var + 2) + n_state_var + 2] \
                                                        for k in range(sequence_length)], axis=-1) \
                                                        for j in range (j_range)] 

            
                state_seq_whole.extend(state_seq_case)
                vel_seq_whole.extend(vel_seq_case)

    state_seq_whole = np.concatenate(state_seq_whole, axis=0)
    vel_seq_whole = np.concatenate(vel_seq_whole, axis=0)
    
    return state_seq_whole, vel_seq_whole

# Normalization
def data_normalization(input_data,no_of_channel):
    norm_data = np.zeros(input_data.shape)
    min_val = []
    max_val = []
    
    # loop through each channel to normalize each index of the channel data
    for i in range(no_of_channel):
        # [batch, height, width, channel]
        # [:: refers to skipping by no_of_channels]
        norm_data[:,:,:,i::no_of_channel] = ((input_data[:,:,:,i::no_of_channel] - np.amin(input_data[:,:,:,i::no_of_channel])) / (np.amax(input_data[:,:,:,i::no_of_channel]) - np.amin(input_data[:,:,:,i::no_of_channel])) + 1E-9)
        min_val.append(np.amin(input_data[:,:,:,i::no_of_channel]))
        max_val.append(np.amax(input_data[:,:,:,i::no_of_channel]))
    return norm_data, min_val, max_val

def data_normalization_test(input_data, min_val, max_val, no_of_channel):
    norm_data = np.zeros(input_data.shape)
    for i in range(no_of_channel):
        norm_data[:,:,:,i::no_of_channel] = ((input_data[:,:,:,i::no_of_channel] - min_val[i]) / (max_val[i] - min_val[i] + 1E-9))
        
    return norm_data

def data_denormalization(input_data, min_val, max_val, no_of_channel):
    denorm_data = np.zeros(input_data.shape)
    for i in range(no_of_channel):
        denorm_data[:,:,:,i::no_of_channel] = (input_data[:,:,:,i::no_of_channel] * (max_val[i] - min_val[i] + 1E-9)) + min_val[i]
    return denorm_data