# utils.py

import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import IPython.display as display
import os
import shutil
import random
import pandas as pd

def create_gif_from_numpy_array(file_path, save_path, interval=5, num_frames=50, duration=100, display_gif=True):
    """
    Create a GIF animation from a numpy array and optionally display it.

    Args:
        file_path (str): Path to the numpy file.
        save_path (str): Path to save the GIF.
        interval (int): Interval between frames in the numpy array.
        num_frames (int): Number of frames to include in the GIF.
        duration (int): Duration of each frame in milliseconds.
        display_gif (bool): Whether to display the GIF after creation.
    """
    # Load the numpy file
    file = np.load(file_path)
    coupled_field_snapshot_check = np.array(file)

    # Initialize list to store frames
    frames = []

    for i in range(num_frames):
        # Plot the image
        plt.imshow(coupled_field_snapshot_check[:, :, i * interval])
        plt.axis('off')  # Turn off axis

        # Convert the figure to PIL image
        plt.subplots_adjust(left=0, right=1, top=1, bottom=0)  # Remove whitespace
        fig = plt.gcf()
        fig.canvas.draw()
        pil_image = Image.frombytes('RGB', fig.canvas.get_width_height(), fig.canvas.tostring_rgb())

        # Close the plot to prevent display
        plt.close()

        # Append the PIL image to the frames list
        frames.append(pil_image)

    # Save frames as a GIF
    frames[0].save(save_path, save_all=True, append_images=frames[1:], duration=duration, loop=0)
    
    # Display the GIF if specified
    if display_gif:
        display.display(display.Image(filename=save_path))
        
def split_data_into_train_test(folder_path, training_path, testing_path, train_ratio=0.8, seed=42):
    """
    Split data into training and testing directories.

    Args:
        folder_path (str): Path to the folder containing the data.
        training_path (str): Path to the training directory.
        testing_path (str): Path to the testing directory.
        train_ratio (float): Ratio of data to be used for training (default is 0.8).
        seed (int): Seed for random number generation (default is 42).
    """
    # Set the random seed
    random.seed(seed)

    # Create the training and testing directories if they don't exist
    os.makedirs(training_path, exist_ok=True)
    os.makedirs(testing_path, exist_ok=True)

    # Get the list of files in the folder
    files = os.listdir(folder_path)

    # Calculate the number of files for training and testing
    num_train = int(len(files) * train_ratio)
    num_test = len(files) - num_train

    # Shuffle the list of files to randomly select for training and testing
    random.shuffle(files)

    # Copy files to training directory
    for file_name in files[:num_train]:
        src = os.path.join(folder_path, file_name)
        dst = os.path.join(training_path, file_name)
        shutil.copy(src, dst)

    # Copy files to testing directory
    for file_name in files[num_train:]:
        src = os.path.join(folder_path, file_name)
        dst = os.path.join(testing_path, file_name)
        shutil.copy(src, dst)

    print("Data split into training and testing directories successfully!")

    
    
def process_file(filepath):
    """
    Process a single file.
    """
    # Read data from file
    read_data = np.genfromtxt(filepath, invalid_raise=False, missing_values=None)

    df = pd.DataFrame(read_data, columns=['X', 'Y', 'U', 'V', 'P', 'T', 'Ls'])
    df = df.drop(['Ls'], axis=1)

    # Process temperature field
    temperature_snapshot_raw = df.pivot_table(index='X', columns='Y', values='T').T.values
    temperature_snapshot = pd.DataFrame(temperature_snapshot_raw).round(6).to_numpy()
    temperature_snapshot = np.expand_dims(temperature_snapshot, axis=2)

    # Process pressure field
    pressure_snapshot_raw = df.pivot_table(index='X', columns='Y', values='P').T.values
    pressure_snapshot = pd.DataFrame(pressure_snapshot_raw).round(6).to_numpy()
    pressure_snapshot = np.expand_dims(pressure_snapshot, axis=2)

    # Process microstructure field
    microstructure_snapshot = temperature_snapshot

    # Process velocity_x field
    vx_snapshot_raw = df.pivot_table(index='X', columns='Y', values='U').T.values
    vx_snapshot = pd.DataFrame(vx_snapshot_raw).round(6).to_numpy()
    vx_snapshot = np.expand_dims(vx_snapshot, axis=2)

    # Process velocity_y field
    vy_snapshot_raw = df.pivot_table(index='X', columns='Y', values='V').T.values
    vy_snapshot = pd.DataFrame(vy_snapshot_raw).round(6).to_numpy()
    vy_snapshot = np.expand_dims(vy_snapshot, axis=2)

    # Combine all snapshots
    coupled_field_snapshot = np.concatenate([temperature_snapshot,
                                              pressure_snapshot,
                                              microstructure_snapshot,
                                              vx_snapshot,
                                              vy_snapshot], axis=2)

    # Cut off temperature to have value from 300 - max temp
    coupled_field_snapshot[:, :, 0][coupled_field_snapshot[:, :, 0] < 300] = 300
    coupled_field_snapshot[:, :, 0][coupled_field_snapshot[:, :, 0] > 5000] = 7000

    # Convert microstructure field to binary image
    coupled_field_snapshot[:, :, 2][coupled_field_snapshot[:, :, 2] > 280] = 280
    coupled_field_snapshot[:, :, 2][coupled_field_snapshot[:, :, 2] < 280] = 0

    return coupled_field_snapshot


def process_subdirectory(subdirectory, np_directory):
    """
    Process all files in a subdirectory.
    """

    # Loop through each subdirectory in the subdirectory
    for inner_subdirectory in os.listdir(subdirectory):
        inner_subdirectory_path = os.path.join(subdirectory, inner_subdirectory)

        # Check if the path is a directory
        if os.path.isdir(inner_subdirectory_path):
            # List all files in the inner subdirectory
            files = os.listdir(inner_subdirectory_path)

            # Extract ts from each file name and sort files based on ts
            files.sort(key=lambda x: int(x.split('_')[-1].split('.')[0][2:]))

            # hold the data
            coupled_field_evolution = []

            # Loop through each file in the inner subdirectory
            for filename in files:
                filepath = os.path.join(inner_subdirectory_path, filename)

                # Check if the path is a file
                if os.path.isfile(filepath):
                    # Process the file and append the snapshot to the list
                    snapshot = process_file(filepath)
                    coupled_field_evolution.append(snapshot)

            coupled_field_evolution = np.concatenate(coupled_field_evolution, axis=2)
            last_part = subdirectory[subdirectory.rfind('/') + 1:]
            np.save(os.path.join(np_directory, f'{last_part}.npy'), coupled_field_evolution)


def convert_data_to_numpy(root_directory, np_directory):
    """
    Process all subdirectories in the root directory.
    """
    i = 0
    # Loop through each subdirectory in the root directory
    for subdirectory in os.listdir(root_directory):
        i += 1
        subdirectory_path = os.path.join(root_directory, subdirectory)

        # Check if the path is a directory
        if os.path.isdir(subdirectory_path):
            # Process the subdirectory
            process_subdirectory(subdirectory_path, np_directory)

        print("File number", i, "complete")
