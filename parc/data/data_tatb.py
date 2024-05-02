import time
import os
import numpy as np
import skimage
from skimage.measure import block_reduce

### idx range is the file range
### sequence length is 
### number of state variables.. maybe temp, pressure, micros?
### purpose is the mde being used for this scenario 
  
def clip_raw_data(idx_range, sequence_length=2, n_state_var=3, purpose = "diff_training"):
    state_seq_whole = []
    vel_seq_whole = []

   # Specify the path to your folder containing the files
    folder_path = '/scratch/jtb3sud/tatb_np'

    # Loop through each file in the folder
    for filename in os.listdir(folder_path):
        if filename.endswith(".npy"):
            file_path = os.path.join(folder_path, filename)
            print(f"Processing {filename}")
            raw_data = np.float32(np.load(file_path))
            data_shape = raw_data.shape
            if data_shape[2] > sequence_length:
                ## padding was for equal size data
                # i do not have this issue 
                #npad = ((0, abs(data_shape[0] - 512)), (0, abs(data_shape[1] - 1024)), (0, 0))
                #raw_data = np.pad(raw_data, pad_width=npad, mode='edge')
                raw_data = np.expand_dims(raw_data, axis=0)
                raw_data = skimage.measure.block_reduce(raw_data[:,:,:,:], (1,4,4,1),np.max)

                data_shape = raw_data.shape
                num_time_steps = data_shape[-1] // (n_state_var + 2)
                if purpose == "diff_training":
                    j_range = num_time_steps - sequence_length
                else:
                    j_range = 1
                ### changing 128 x 208
                state_seq_case = [np.concatenate([raw_data[:, :128, :208, (j + k) * (n_state_var + 2):\
                                                        (j + k) * (n_state_var + 2) + n_state_var] \
                                                        for k in range(sequence_length)], axis=-1) \
                                                        for j in range  (j_range)] 

                vel_seq_case = [np.concatenate([raw_data[:, :128, :208, (j + k) * (n_state_var + 2) +  n_state_var :\
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
    for i in range(no_of_channel):
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