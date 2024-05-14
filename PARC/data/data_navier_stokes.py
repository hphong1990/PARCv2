import os
from parc.data.base_data import BaseData
import numpy as np
import skimage
from skimage.measure import block_reduce

class DataNS(BaseData):
    def __init__(self, Re_list, **kwargs, ):
        super(DataNS, self).__init__(**kwargs)
        self.Re_list = Re_list
        # Download data

    def information(self):
        print("Train ")
        pass
    
    def import_data(self, folder_path):

        """
        This function is to import all of NS data
        Input: folder_path
        Output: numpy array in the form B x H x W x C
        """
        data_whole = []
        # r_whole = []
        for Re in self.Re_list: # Loop through the file list
            data_file_name = 'Re_' + str(int(Re)) + '.npy'
            file_path = folder_path + data_file_name                
            if os.path.exists(file_path): # If file exists
                # Load raw data
                raw_data = np.float32(np.load(file_path))

                # Reshaping data to have B x H x W x C format
                raw_data = np.expand_dims(raw_data, axis = 0)
                data_whole.extend(raw_data)

        # Combine data list to numpy array
        data_whole = np.concatenate([data_whole], axis=0)
        return data_whole

    def train_test_split(self, Re_train, Re_test, Re_list, input_data):

        """
        This function is to split data to train, test. 
        Input: 
        - Re_train (list): reynold number for training
        - Re_test (list): reynold number for testing
        - Re_list (list): list of all reynold number
        - input_data (numpy array (B x H x W x C)): data in numpy format
        """
        idx = 0
        train_idx =[]
        test_idx =[]
        # Loop through Re_list and retrieve test and train reynold number
        for Re in Re_list:
            if Re in Re_train:
                train_idx.append(idx)
            elif Re in Re_test:
                test_idx.append(idx)
            idx += 1
        # Extract training data
        train_seq = [input_data[idx:idx+1,:,:,:] for idx in train_idx]

        # Extract testing data
        test_seq = [input_data[idx:idx+1,:,:,:] for idx in test_idx]

        # Make training array
        train_seq = np.concatenate(train_seq, axis = 0)

        # Make testing array
        test_seq = np.concatenate(test_seq, axis = 0)
        return train_seq, test_seq

    def clip_data(self, input_seq, no_of_fields, sequence_length = 2):
        """
        Make training and testing sequence
        Input:
        - input_seq: input data with full length
        - no_of_fields: number of state/velocity variables
        - sequence_length: the length of clipping sequence
        """
        # Get input shape
        shape = input_seq.shape
        num_time_steps = np.int32((shape[-1]-1)/3)
        vel_seq_whole = []
        for i in range(shape[0]):
            for j in range(num_time_steps-sequence_length+1):
                vel_seq_case = np.expand_dims(input_seq[i, :, :, (j*no_of_fields):(j*no_of_fields+sequence_length*no_of_fields)],axis = 0)
                vel_seq_whole.extend(vel_seq_case)
        vel_seq_whole = np.concatenate([vel_seq_whole], axis=0)
        return vel_seq_whole
    
# train_data = create_train_data(train_seq, no_of_fields = 3, sequence_length = 13)
