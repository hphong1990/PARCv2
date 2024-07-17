import numpy as np
import tensorflow as tf
import os
import glob
from skimage.measure import block_reduce


def dat2npy(metalearning_dir, species, data_shape=(600, 1000), timesteps=60, 
            timeskips=1000, temp_clip=[300, 5000],
            filename_patterns=["xyuvpTLs_ts%i.dat", "xyuvpTLs_ts%07i.dat"]):
    data_dir = os.path.join(metalearning_dir, species)
    simulations_dir = glob.glob(os.path.join(data_dir, "*"))
    for sim_dir in simulations_dir:
        if not os.path.isdir(sim_dir):
            continue
        sim_id = os.path.basename(os.path.normpath(sim_dir))
        time_steps = [1]
        time_steps.extend([i * timeskips for i in range(1, timesteps)])
        # Start reading the files
        simulation_frames = []
        for ts in time_steps:
            for each_pattern in filename_patterns:
                try:
                    snapshot_dir_to_grab = os.path.join(sim_dir, "*", each_pattern % ts)
                    snapshot_dir = list(glob.glob(snapshot_dir_to_grab))[0]
                    snapshot_data = np.loadtxt(snapshot_dir)
                    break
                except Exception as e:
                    continue
            # T p mu u v
            snapshot = np.empty((data_shape[0], data_shape[1], 5), dtype=np.float32)
            snapshot[:, :, 0] = snapshot_data[:, 5].reshape(data_shape[0], data_shape[1])
            if temp_clip is not None:
                 np.clip(snapshot[:, :, 0], temp_clip[0], temp_clip[1], out=snapshot[:, :, 0])
            snapshot[:, :, 1] = snapshot_data[:, 4].reshape(data_shape[0], data_shape[1])
            snapshot[:, :, 2] = snapshot_data[:, 5].reshape(data_shape[0], data_shape[1])
            snapshot[:, :, 3] = snapshot_data[:, 2].reshape(data_shape[0], data_shape[1])
            snapshot[:, :, 4] = snapshot_data[:, 3].reshape(data_shape[0], data_shape[1])
            simulation_frames.append(snapshot)
        simulation_frames = np.array(simulation_frames)
        simulation_frames = block_reduce(simulation_frames, (1, 4, 4, 1), np.max)
        simulation_frames = simulation_frames[:, 3:-3, :240, :]
        np.save(os.path.join(data_dir, sim_id + ".npy"), simulation_frames)
        
        
def load_tfdataset(simnpy_list, seq=2, n_state_var=3):
    for idx, each_sim in enumerate(simnpy_list):
        tmp = np.load(each_sim)
        tfds_tmp = tf.data.Dataset.from_tensor_slices(tmp)
        tfds_tmp = tfds_tmp.window(seq, shift=1, drop_remainder=True)
        tfds_tmp = tfds_tmp.flat_map(lambda window: window.batch(seq))
        tfds_tmp = tfds_tmp.map(lambda window: 
                                ((window[0, :, :, :n_state_var], window[0, :, :, n_state_var:]),
                                 (window[1:, :, :, :n_state_var], window[1:, :, :, n_state_var:])))
        if idx == 0:
            tfds = tfds_tmp
        else:
            tfds = tfds.concatenate(tfds_tmp)
    return tfds