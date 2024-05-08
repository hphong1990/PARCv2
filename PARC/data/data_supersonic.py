import os
from parc.data.base_data import BaseData
import numpy as np
import tensorflow as tf
from skimage.measure import block_reduce


class DataSupersonic(BaseData):
    def __init__(self, **kwargs):
        super(DataSupersonic, self).__init__(**kwargs)
        self.data_shape = (500, 750, 4)
        # Hardcoded min/max from dataset as the dataset is too large to be read into memory at once
        self.data_min = np.array([0.0, 0.0, -3.0178108, -3.736057])
        self.data_max = np.array([13.75365, 147.34137, 5.5204816, 3.7361536])
        # TODO: Download and storage
        self.data_dir = os.path.join("data", "supersonic")
        self.mach_num_list = [1.1, 1.2, 1.3, 1.4, 1.5, 1.6, 1.7, 1.8, 2.0, 
                              2.1, 2.2, 2.3, 2.4, 2.5, 2.6, 2.7, 2.8, 2.9, 3.0, 
                              3.1, 3.2, 3.3, 3.4, 3.5, 3.6, 3.7, 3.8, 3.9, 4.0, 
                              4.1, 4.2, 4.3, 4.4, 4.5, 4.6, 4.7, 4.8, 4.9, 5.0]
    
    def read_raw_mach_number(self, mach_number, start_frame):
        '''
        Read raw simulation data for each mach number. Assume EM data model and structure.
        
        Parameters
        ---
        mach_number: float, the mach number of the simulation to read
        start_frame: int, the starting frame of the simulation
        
        Returns
        ---
        simulation_frames: 4-d np.array, the simulation data in [t, x, y, c] format with c=[rho, p, u, v]
        '''
        mach_number_dir = os.path.join(self.data_dir, "Mach_%.1f_long" % mach_number)
        frame_time_array = np.loadtxt(os.path.join(mach_number_dir, "VoidCollapse.contour.timesteps"))
        simulation_frames = []
        for each_frame in frame_time_array:
            if each_frame[0] < start_frame:
                continue
            try:
                frame_snapshot = np.loadtxt(os.path.join(mach_number_dir, "xyuvpRho_ts%07i.txt" % each_frame[0]))
            except Exception as e:
                print(str(mach_number), str(each_frame[0]), str(e))
                continue
            snapshot = np.empty(self.data_shape, dtype=np.float32)
            # Rho, p, u, v
            snapshot[:, :, 0] = frame_snapshot[:, 5].reshape(self.data_shape[0], self.data_shape[1])
            snapshot[:, :, 1] = frame_snapshot[:, 4].reshape(self.data_shape[0], self.data_shape[1])
            snapshot[:, :, 2] = frame_snapshot[:, 2].reshape(self.data_shape[0], self.data_shape[1])
            snapshot[:, :, 3] = frame_snapshot[:, 3].reshape(self.data_shape[0], self.data_shape[1])
            simulation_frames.append(snapshot)
        simulation_frames = np.array(simulation_frames)
        # Downsample by factor of 4
        simulation_frames = block_reduce(simulation_frames, (1, 4, 4, 1), np.mean)
        # Multiple of 16
        simulation_frames = simulation_frames[:, 6:-7, :176, :]
        return simulation_frames
    
    def normalize_and_save(self, mach_num_list=self.mach_num_list):
        '''
        Normalize data and save. Note that the min and max are hardcoded due to the size of the raw data. 
        Normalized data will be saved as intermediate npy files for later usage.
        
        Parameters
        ---
        mach_num_list: list, default self.mach_num_list, the list of mach number to process
        
        Returns
        ---
        None
        '''
        for each_mach in mach_number_list:
            hypersonic_data = read_raw_mach_number(self.data_dir, each_mach, 1)
            # Normalization
            hypersonic_data = (hypersonic_data - self.data_min[None, None, None, :]) / (self.data_max[None, None, None, :] - self.data_min[None, None, None, :])
            np.save(os.path.join(self.data_dir, "normalized", "mach_%.1f.npy" % each_mach), hypersonic_data)
            print("Mach %.1f: %i" % (each_mach, hypersonic_data.shape))
    
    def load_tfdataset(self, mach_num_list, seq=2, time_downsample=3):
        '''
        Load dataset with tf.data.Dataset and return training sample of given length. This method only loads data when necessary, thus avoiding excessive memory usage.
        
        Parameters
        ---
        mach_num_list: list, the list of mach number to process
        seq: int, default 2, the total sequence length for training/testing
        time_downsample: int, default 3, factor of temporal downsample
        
        Returns
        ---
        tfds: tf.data.Dataset, the dataset ready to use. Each sample in tfds is a (2,2) tuple of np.array. See the following table for data shape and data model
              |---   |---      |---        |
              |index |shape    |description|
              |---   |---      |---        |
              |[0][0]|(112, 176, 2)|initial state variables|
              |[0][1]|(112, 176, 2)|initial velocity variables|
              |[1][0]|(seq-1, 112, 176, 2)|sequence of subsequent state variables|
              |[1][0]|(seq-1, 112, 176, 2)|sequence of subsequent velocity variables|
              |---   |---      |---        |
        '''
        for idx, each_mach in enumerate(mach_num_list):
            tmp = np.load(os.path.join(self.data_dir, "normalized", "mach_%.1f.npy" % each_mach))
            if time_downsample != 1:
                tmp = tmp[::time_downsample, :, :, :]
            tfds_tmp = tf.data.Dataset.from_tensor_slices(tmp)
            tfds_tmp = tfds_tmp.window(seq, shift=1, drop_remainder=True)
            tfds_tmp = tfds_tmp.flat_map(lambda window: window.batch(seq))
            tfds_tmp = tfds_tmp.map(lambda window: 
                                    ((window[0, :, :, :2], window[0, :, :, 2:]),
                                     (window[1:, :, :, :2], window[1:, :, :, 2:])))
            if idx == 0:
                tfds = tfds_tmp
            else:
                tfds = tfds.concatenate(tfds_tmp)
        return tfds