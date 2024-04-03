import os
from parc.data.base_data import BaseData
import numpy as np
import skimage
from skimage.measure import block_reduce

class DataEnergeticMaterials(BaseData):
    def __init__(self, **kwargs):
        super(DataEnergeticMaterials, self).__init__(**kwargs)
        # Download data

    def information(self):
        print("Train ")
        pass

    def clip_raw_data(idx_range, folder_path, sequence_length=2, n_state_var=3, purpose = "diff_training", target_size = (512,1024), dim_reduce = 4):
        # Initiate the numpy array for state and velocity
        state_seq_whole = [] # State
        vel_seq_whole = []  # Velocity

        # Looping through the file list
        for i in range(idx_range[0],idx_range[1]):
            file_path = folder_path + 'void_' + str(i) +'.npy'
            if os.path.exists(file_path):
                # Load the data file
                raw_data = np.float32(np.load(file_path))

                # Padding to make the dimension uniform across different cases
                npad = ((0, abs(raw_data.shape[0] - target_size[0])), (0, abs(raw_data.shape[1] - target_size[1])), (0, 0))
                raw_data = np.pad(raw_data, pad_width=npad, mode='edge')

                # Expand dimension to create the batch dimension
                raw_data = np.expand_dims(raw_data, axis=0)

                # Reduce the spatial dimension
                raw_data = skimage.measure.block_reduce(raw_data[:,:,:,:], (1,dim_reduce,dim_reduce,1),np.max)

                # Caculate the number of time step in the input sequence
                num_time_steps = raw_data.shape[-1] // (n_state_var + 2)

                # Cut the original sequence to training or testing sequence
                # Depend on purpose, sliding window will be used (for training) or not (for testing)
                if purpose == "diff_training":
                    j_range = num_time_steps - sequence_length
                else:
                    j_range = 1
                
                # Extract data for state variables
                state_seq_case = [np.concatenate([raw_data[:, :, :, (j + k) * (n_state_var + 2):\
                                                        (j + k) * (n_state_var + 2) + n_state_var] \
                                                        for k in range(sequence_length)], axis=-1) \
                                                        for j in range  (j_range)] 
                
                # Extract data for velocity variables
                vel_seq_case = [np.concatenate([raw_data[:, :, :, (j + k) * (n_state_var + 2) +  n_state_var :\
                                                        (j + k) * (n_state_var + 2) + n_state_var + 2] \
                                                        for k in range(sequence_length)], axis=-1) \
                                                        for j in range (j_range)] 
                
                # Append to whole training/testing set
                state_seq_whole.extend(state_seq_case)
                vel_seq_whole.extend(vel_seq_case)
        
        # Create final array
        state_seq_whole = np.concatenate(state_seq_whole, axis=0)
        vel_seq_whole = np.concatenate(vel_seq_whole, axis=0)
        
        return state_seq_whole, vel_seq_whole
        