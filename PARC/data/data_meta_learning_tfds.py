import numpy as np
import tensorflow as tf
import os
import glob
from skimage.measure import block_reduce


def dat2npy(metalearning_dir, species, data_shape=(600, 1000), timesteps=60, 
            timeskips=1000, temp_clip=[300, 5000],
            filename_patterns=["xyuvpTLs_ts%i.dat", "xyuvpTLs_ts%07i.dat"],
            overwrite=False):
    data_dir = os.path.join(metalearning_dir, species)
    simulations_dir = glob.glob(os.path.join(data_dir, "*"))
    for sim_dir in simulations_dir:
        if not os.path.isdir(sim_dir):
            continue
        sim_id = os.path.basename(os.path.normpath(sim_dir))
        if sim_id == "time":
            continue
        if ((not overwrite) and os.path.isfile(os.path.join(data_dir, sim_id + ".npy"))):
            print("Simulation ID %s has been converted and will not be overwritten." % sim_id)
            continue
        print("Begin processing Simulation ID %s" % sim_id)
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
        # Look for time file
        sim_id = os.path.basename(os.path.normpath(each_sim))
        ts = np.load(os.path.normpath(os.path.join(each_sim, "..", "..", "time", sim_id)))
        tfds_ts = tf.data.Dataset.from_tensor_slices(ts).window(seq, shift=1, drop_remainder=True).flat_map(lambda window: window.batch(seq))
        # Simulation data
        tmp = np.load(each_sim)
        tfds_tmp = tf.data.Dataset.from_tensor_slices(tmp).window(seq, shift=1, drop_remainder=True).flat_map(lambda window: window.batch(seq))
        # Zip and split out the first
        tfds_tmp = tf.data.Dataset.zip(tfds_tmp, tfds_ts)
        tfds_tmp = tfds_tmp.map(lambda tmp, ts: 
                                ((tmp[0, :, :, :n_state_var], tmp[0, :, :, n_state_var:], ts[0]),
                                 (tmp[1:, :, :, :n_state_var], tmp[1:, :, :, n_state_var:], ts[1:])))
        if idx == 0:
            tfds = tfds_tmp
        else:
            tfds = tfds.concatenate(tfds_tmp)
    return tfds

# reaction.dat in tatb format
def reaction_dat_tatb(filename, timeskips, timesteps):
    reaction_dat = np.genfromtxt(filename, skip_header=2, usecols=(0, 1))[::4, :]
    # Sanity check
    assert reaction_dat.shape[0] == 60000
    assert reaction_dat.shape[1] == 2
    # Filter out based on timeskips
    timesteps = np.array([1] + [i * timeskips for i in range(1, timesteps)])
    mask = np.isin(reaction_dat[:, 1], timesteps)
    ts = reaction_dat[mask, 0]
    # Average delta_t and min/max deviation
    dt = np.diff(ts)
    ts_mean = np.mean(dt)
    print("delta_t Mean: %.4e Max +%.2f%% Min -%.2f%%" % (ts_mean, (np.max(dt) - ts_mean)/ts_mean * 100.0, (ts_mean - np.min(dt))/ts_mean * 100.0))
    return ts


# reaction.dat in HMX format
def reaction_dat_hmx(filename, timeskips, timesteps):
    reaction_dat = np.genfromtxt(filename, skip_header=1, usecols=(0, 1))
    # Sanity check
    assert reaction_dat.shape[0] == 60000
    assert reaction_dat.shape[1] == 2
    # Filter out based on timeskips
    timesteps = np.array([1] + [i * timeskips for i in range(1, timesteps)])
    mask = np.isin(reaction_dat[:, 1], timesteps)
    ts = reaction_dat[mask, 0]
    # Average delta_t and min/max deviation
    dt = np.diff(ts)
    ts_mean = np.mean(dt)
    print("delta_t Mean: %.4e Max +%.2f%% Min -%.2f%%" % (ts_mean, (np.max(dt) - ts_mean)/ts_mean * 100.0, (ts_mean - np.min(dt))/ts_mean * 100.0))
    return ts


def dat2npy_time(metalearning_dir, species, parsing_func, timesteps=60, timeskips=1000, overwrite=False):
    data_dir = os.path.join(metalearning_dir, species)
    simulations_dir = glob.glob(os.path.join(data_dir, "*"))
    for sim_dir in simulations_dir:
        if not os.path.isdir(sim_dir):
            continue
        sim_id = os.path.basename(os.path.normpath(sim_dir))
        if ((not overwrite) and os.path.isfile(os.path.join(data_dir, sim_id + "_t.npy"))):
            print("Simulation ID %s has been converted and will not be overwritten." % sim_id)
            continue
        print("Begin processing Simulation ID %s Reaction.dat" % sim_id)
        ts = parsing_func(os.path.join(sim_dir, "Reaction.dat"), timeskips, timesteps)
        os.makedirs(os.path.join(data_dir, "time"), exist_ok=True)
        np.save(os.path.join(data_dir, "time", sim_id + "_t.npy"), ts)