import time
import skimage
from skimage.measure import block_reduce

import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
import os
os.chdir(".")
import parc.data.data_em as data
from parc.model import model_em as model

# read in data
state_path = "/usr/workspace/gray65/data/class5_mesoscale_processed/mesoscale_x.npy"
vel_path = "/usr/workspace/gray65/data/class5_mesoscale_processed/mesoscale_u.npy"

state_seq_whole = np.float32(np.load(state_path))
vel_seq_whole = np.float32(np.load(vel_path))

# Calculate the amount of padding needed
target_height = 192
padding_needed = target_height - state_seq_whole.shape[1]

# Calculate padding for the height dimension
padding_top = padding_needed // 2
padding_bottom = padding_needed - padding_top

# Define the padding configuration
paddings = [[0, 0], [padding_top, padding_bottom], [0, 0], [0, 0]]

# Apply padding
state_seq_whole = np.pad(state_seq_whole, paddings, mode='constant', constant_values=0)
vel_seq_whole = np.pad(vel_seq_whole, paddings, mode='constant', constant_values=0)

# normalize
state_seq_norm = data.data_normalization(state_seq_whole,3)
vel_seq_norm = data.data_normalization(vel_seq_whole,2)

# TRAINING

# Create tf.dataset
dataset_input = tf.data.Dataset.from_tensor_slices((state_seq_norm[0][:,:,:,:3],vel_seq_norm[0][:,:,:,:2]))
dataset_label = tf.data.Dataset.from_tensor_slices((state_seq_norm[0][:,:,:,-3:],vel_seq_norm[0][:,:,:,-2:]))
dataset = tf.data.Dataset.zip((dataset_input, dataset_label))
dataset = dataset.shuffle(buffer_size = 2192) 
dataset = dataset.batch(4) # changed to 4 from 8 because data too big for 8

tf.keras.backend.clear_session()
parc = model.PARCv2(n_state_var = 3, n_time_step = 1, step_size= 1/15, solver = "rk4", mode = "differentiator_training")
# parc.differentiator.load_weights('class5_mse_1200epoch.h5')
parc.compile(optimizer = tf.keras.optimizers.Adam(learning_rate = 0.00001, beta_1 = 0.9, beta_2 = 0.999))
parc.fit(dataset, epochs = 10, shuffle = True)

parc.differentiator.save_weights('/usr/workspace/gray65/job_outputs/lassen_test_weights.h5')

# Validation

# Load Weights
tf.keras.backend.clear_session()
parc_rk = model.PARCv2(n_state_var = 3, n_time_step = 14, step_size= 1/15, solver = "rk4")
parc_rk.compile()
parc_rk.differentiator.load_weights('/usr/workspace/gray65/job_outputs/lassen_test_weights.h5')

idx = 0
state_var_init = tf.cast(state_seq_norm[0][idx:idx+1,:,:,:3], tf.float32)
velocity_init = tf.cast(vel_seq_norm[0][idx:idx+1,:,:,:2], tf.float32)
input_seq_current = tf.concat([state_var_init, velocity_init],axis = -1)

state_whole = []
vel_whole = []
for ts in range(30):
    input_seq_current, update = parc_rk.explicit_update(input_seq_current)
    state_whole.append(input_seq_current[:,:,:,:3])
    vel_whole.append(input_seq_current[:,:,:,3:])
state_pred = tf.concat(state_whole, axis = -1).numpy()
vel_pred = tf.concat(vel_whole, axis = -1).numpy()

for i in range(30):
    fig, ax = plt.subplots(1,2) # switch to (1,2) for horizontal
    ax[0].imshow(state_seq_whole[i+idx, :, :, 3])
    ax[1].imshow(state_pred[0, :, :, i * 3], vmin=0, vmax=1)
    if i % 5 == 0 or i == 29:
        plt.savefig(f'/usr/workspace/gray65/job_outputs/lassen_test_10epoch{i}.png')
        plt.show()
    plt.close(fig)  # Close the figure after each iteration to free memory
