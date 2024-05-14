import os
from parc.data.base_data import BaseData
import numpy as np
import skimage
from skimage.measure import block_reduce

class DataBurgers(BaseData):
    def __init__(self, **kwargs):
        super(DataBurgers, self).__init__(**kwargs)
        # Download data

    def information(self):
        print("Train ")
        pass

    def clip_raw_data(self, R_list, a_list, w_list, sequence_length=2, purpose = 'training'):
        vel_seq_whole = []
        if purpose == 'training':
            base_name = 'burgers_train_'
            base_multiplier = 10
            folder_path = './data/burgers/train/'
        else:
            base_name = 'burgers_test_'
            base_multiplier = 100
            folder_path = './data/burgers/test/'
        # Looping through the file list
        for R in R_list: # Loop through Re number list
            for a in a_list:    # Loop through signal magnitude list
                for w in w_list:    # Loop through signal width list
                    data_file_name = base_name + str(int(R)) + '_' + str(int(a*base_multiplier)) + '_' + str(int(w*base_multiplier)) + '.npy'
                    file_path = folder_path + data_file_name                
                    if os.path.exists(file_path): # Check if path exist
                        # Load data
                        raw_data = np.float32(np.load(file_path))
                        # Reorganize tensor shape
                        raw_data = np.moveaxis(raw_data,-2,0)
                        data_shape = raw_data.shape
                        # Compute the number of steps
                        num_time_steps = data_shape[0]

                        # Create constant tensor
                        norm_r = R/15000
                        r_img = norm_r*np.ones(shape = (1,data_shape[1],data_shape[2],1))
                        
                        # Reorganize tensor shape
                        if purpose == 'training':
                            looping_range = num_time_steps-sequence_length
                        else:
                            looping_range = 1
                        for j in range (looping_range):
                            # Assemble first step
                            init_snapshot = np.concatenate([raw_data[j:j+1, :, :, :],r_img],axis = -1)

                            # Collect the rest
                            following_snapshot = []
                            for k in range(sequence_length-1):
                                following_snapshot.append(raw_data[(j + k +1):(j + k + 2), :, :, :])
                            following_snapshot = np.concatenate(following_snapshot,axis = -1)
                            # Assemble all
                            vel_seq_case = np.concatenate([init_snapshot,following_snapshot],axis = -1)
                            vel_seq_whole.append(vel_seq_case)
        # Make data tensor
        vel_seq_whole = np.concatenate(vel_seq_whole, axis=0)
        return vel_seq_whole
    

