import os
from .base_data import BaseData
import numpy as np
import skimage
from skimage.measure import block_reduce
import os.path as osp

# todo: makit it class seems unnecessary
class DataEnergeticMaterials(BaseData):
    def __init__(self, **kwargs):
        super(DataEnergeticMaterials, self).__init__(**kwargs)
        # Download data
        # todo: make it downloadable

    # todo: commeent
    # todo: not needed? 
    def information(self):
        print("Train ")
        pass

    def clip_raw_data(self, dataset_range, dir_dataset, n_seq=2, n_state=3, use_sldg_wdn = True, tgt_sz = (512,1024), dim_reduce = 4):
        """ 
        Process single void simulation data to construct dataset. 
        
        It takes single void simulations in their individual numpy files with the following notes: 
            1) each simulation has a different number of time steps
            2) in the format of (1, X, Y, timestep + state + velocity)
            3) From raw data, Temperature has been clipped to [300, 5000], microstructure has been binarized, and Pressure untouched
        Args:
            dataset_range: (tuple) range of void cases to include
            dir_dataset: (str) directory containing void simulations
            n_seq: (int) number of timesteps for sequence to consider, i.e., n_seq=2 yield sample t_i and t_i+1
            n_state: (int) number of state variables, (def=3: temperature, pressure, and microstructure)
            use_sldg_wdn: (bool) indicating if a sliding window is used to clip time sequence
            tgt_sz: (int, int) output spatial dimension
            dim_reduce: (int) factor of downsampling
        Returns:
            X_dataset (numpy): (cases + timesteps, X, Y, state * n_seq) state variable 
            U_dataset (numpy): (cases + timesteps, X, Y, velocity * n_seq) velocity variable 
        """

        X_dataset, U_dataset = [], []
        n_dataset = 0

        for case_idx in range(dataset_range[0], dataset_range[1]): 
            dir_case = osp.join( dir_dataset, 'void_' + str(case_idx) + '.npy' )
            if osp.exists( dir_case ):
                n_dataset += 1
                data = np.float32( np.load( dir_case ) ) # (X, Y, ts + state + velocity)
                n_ts = data.shape[-1] // (n_state + 2)

                """ pad to target size and downsample """ 
                npad = (
                    (0, abs(data.shape[0] - tgt_sz[0])), 
                    (0, abs(data.shape[1] - tgt_sz[1])), 
                    (0,0))
                data = np.pad(data, pad_width=npad, mode='edge')
                data = np.expand_dims(data, axis=0) # (1, X, Y, ts + state + velocity)

                data = skimage.measure.block_reduce(data, (1, dim_reduce, dim_reduce, 1), np.max)

                ts = n_ts - n_seq if use_sldg_wdn else 1 # starting index range for sequence data

                """ extract X and U """ 
                X_extracted = [None] * ts
                for ti in range(ts): # loop through different timesteps
                    seq = [None] * n_seq
                    for seq_i in range(n_seq): # loop through number of sequence
                        st = (ti + seq_i) * (n_state + 2)
                        end = st + n_state
                        seq[seq_i] = data[..., st:end]
                    seq = np.concatenate( seq, axis=-1 )
                    X_extracted[ti] = seq

                U_extracted = [None] * ts
                for ti in range(ts):
                    seq = [None] * n_seq
                    for seq_i in range(n_seq):
                        st = (ti + seq_i) * (n_state + 2) + n_state
                        end = st + 2
                        seq[seq_i] = data[..., st:end]
                    seq = np.concatenate( seq, axis=-1 )
                    U_extracted[ti] = seq

                X_dataset.extend( X_extracted ) 
                U_dataset.extend( U_extracted )
        X_dataset = np.concatenate(X_dataset, axis=0)
        U_dataset = np.concatenate(U_dataset, axis=0)
        print( f"Processed {n_dataset} simulation data into State {X_dataset.shape} and Velocity {U_dataset.shape}" )
        return X_dataset, U_dataset