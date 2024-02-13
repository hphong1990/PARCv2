import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
############
def plot_latent_prediction(state, state_var_type, initial_ts):
    ylabel_ls = ['GT', 'Latent_PARC'] 
    if state_var_type == "temperature":
        min_val = 300  # min temperature (K)
        max_val = 4000  # max temperature (K)
        unit = "(K)"
        # plot the prediction results
        fig, ax = plt.subplots(2, 5, figsize=(20, 4))
        plt.subplots_adjust(wspace=0.06, hspace=0.07, top=0.85)
        for i in range(2):
            for j in range(5):
                ax[i][j].clear()
                ax[i][j].set_xticks([])
                ax[i][j].set_yticks([])
                ax[i][j].imshow(np.squeeze(state[i, j+initial_ts, :, :]), cmap='jet', vmin=-1, vmax=1)
            ax[i][0].set_ylabel(ylabel_ls[i], fontsize=15)    
        fig.subplots_adjust(right=0.8)
        cbar_ax = fig.add_axes([0.81, 0.125, 0.015, 0.725])
        norm = mpl.colors.Normalize(vmin=min_val, vmax=max_val)
        cbar = fig.colorbar(
            mpl.cm.ScalarMappable(norm=norm, cmap="jet"), cax=cbar_ax, ax=ax[0][0]
        )
        cbar.set_label(label=unit, fontsize=20)
        cbar.ax.tick_params(labelsize=20)
    ########
    elif state_var_type == "pressure":
        (s1, s2) = (np.amin(state[0,:]), np.amax(state[0,:]))
        (s_min, s_max) = ((26*s1 + 24), (26*s2 + 24)) # convert back the normalized pressure to the original range [-2,50] GPa
        # plot the prediction results
        fig, ax = plt.subplots(2, 5, figsize=(20, 4))
        plt.subplots_adjust(wspace=0.06, hspace=0.07, top=0.85)
        for i in range(2):
            for j in range(5):
                ax[i][j].clear()
                ax[i][j].set_xticks([])
                ax[i][j].set_yticks([])
                ax[i][j].imshow(np.squeeze(state[i, j+initial_ts, :, :]), cmap='jet', vmin=s1, vmax=s2)
            ax[i][0].set_ylabel(ylabel_ls[i], fontsize=15)  
        fig.subplots_adjust(right=0.8)
        cbar_ax = fig.add_axes([0.81, 0.125, 0.015, 0.725])
        norm = mpl.colors.Normalize(vmin=s_min, vmax=s_max)
        cbar = fig.colorbar(
            mpl.cm.ScalarMappable(norm=norm, cmap="jet"), cax=cbar_ax, ax=ax[0][0]
        )
        cbar.set_label(label='GPa', fontsize=20)
        cbar.ax.tick_params(labelsize=20)
    elif state_var_type == "microstructure":
        fig, ax = plt.subplots(2, 5, figsize=(20, 4))
        plt.subplots_adjust(wspace=0.06, hspace=0.07, top=0.85)
        for i in range(2):
            for j in range(5):
                ax[i][j].clear()
                ax[i][j].set_xticks([])
                ax[i][j].set_yticks([])
                ax[i][j].imshow(np.squeeze(state[i, j+initial_ts, :, :]), vmin=-1, vmax=1)
            ax[i][0].set_ylabel(ylabel_ls[i], fontsize=15)  
    return fig