import numpy as np
import tensorflow as tf
import os
import glob
from skimage.measure import block_reduce


def dat2npy(metalearning_dir, species, data_shape=(600, 1000), timesteps=60, timeskips=1000):
    data_dir = os.path.join(metalearning_dir, species, "data")
    simulations_dir = glob.glob(os.path.join(data_dir, "*"))
    for sim_dir in simulations_dir:
        if not os.path.isdir(sim_dir):
            continue
        sim_id = os.path.basename(os.path.normpath(sim_dir))
        time_steps = [1]
        time_steps.extend([i * timeskips for i in range(1, timesteps)])
        simulation_frames = []
        for ts in time_steps:
            try:
                snapshot_dir = os.path.join(sim_dir, "xyuvpTLs_ts%i.dat" % ts)
                snapshot_data = np.loadtxt(snapshot_dir)
            except Exception as e:
                print(str(e))
                continue
            # T p mu u v
            snapshot = np.empty((data_shape[0], data_shape[1], 5), dtype=np.float32)
            snapshot[:, :, 0] = snapshot_data[:, 5].reshape(data_shape[0], data_shape[1])
            snapshot[:, :, 1] = snapshot_data[:, 4].reshape(data_shape[0], data_shape[1])
            snapshot[:, :, 2] = snapshot_data[:, 5].reshape(data_shape[0], data_shape[1])
            snapshot[:, :, 3] = snapshot_data[:, 2].reshape(data_shape[0], data_shape[1])
            snapshot[:, :, 4] = snapshot_data[:, 3].reshape(data_shape[0], data_shape[1])
            simulation_frames.append(snapshot)
            print(ts)
        simulation_frames = np.array(simulation_frames)
        simulation_frames = block_reduce(simulation_frames, (1, 4, 4, 1), np.max)
        simulation_frames = simulation_frames[:, 3:-3, :240, :]
        np.save(os.path.join(data_dir, sim_id + ".npy"), simulation_frames)
        
        
def load_tfdataset(simnpy_list, seq=2):
    for idx, each_sim in enumerate(simnpy_list):
        tmp = np.load(each_sim)
        tfds_tmp = tf.data.Dataset.from_tensor_slices(tmp)
        tfds_tmp = tfds_tmp.window(seq, shift=1, drop_remainder=True)
        tfds_tmp = tfds_tmp.flat_map(lambda window: window.batch(seq))
        tfds_tmp = tfds_tmp.map(lambda window: 
                                ((window[0:1, :, :, :3], window[0:1, :, :, 3:]),
                                 (window[1:, :, :, :3], window[1:, :, :, 3:])))
        if idx == 0:
            tfds = tfds_tmp
        else:
            tfds = tfds.concatenate(tfds_tmp)
    return tfds