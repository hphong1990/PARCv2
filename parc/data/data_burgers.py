import os
from parc.data.data_base import BaseData
import numpy as np
import skimage
from skimage.measure import block_reduce

class DataBurgers(BaseData):
    def __init__(self, **kwargs):
        super(DataBurgers, self).__init__(**kwargs)

    def clip_raw_data(folder_path, R_list, a_list, w_list, sequence_length=2, purpose = 'training'):
        # Download data

        
        vel_seq_whole = []
        # Looping through the file list
        for R in R_list:
            for a in a_list:
                for w in w_list:
                    data_file_name = 'burgers_train_' + str(int(R)) + '_' + str(int(a*10)) + '_' + str(int(w*10)) + '.npy'
                    file_path = folder_path + data_file_name                
                    if os.path.exists(file_path):
                        # Load data
                        raw_data = np.float32(np.load(file_path))
                        # Reorganize tensor shape
                        raw_data = np.moveaxis(raw_data,-2,0)
                        data_shape = raw_data.shape
                        num_time_steps = data_shape[0]
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

        vel_seq_whole = np.concatenate(vel_seq_whole, axis=0)
        return vel_seq_whole
    

