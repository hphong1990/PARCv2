import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
import tensorflow as tf
from scipy.ndimage import gaussian_filter
from PIL import ImageFilter, Image
from IPython.display import display

def plot_field_evolution(y_pred, y_true, case_id, options):
    """Visualize the prediction
    :param y_pred:          (numpy) predicted fields
                            [0, ...] predicted fields (F)
                            [1, ...] predicted change of fields (F_dot)
    :param y_true:          (numpy) true label of the fields
    :param test_sample_no:  (int)   array index to select the test case
    :state_var_type:        (str)   indicate which fields to plot the result to apply correct scaling
    """

    # get correct scaling terms
    # opts_2 = 0
    step = 4
    
    if options == "temperature":
        
        min_val = 300  # min temperature (K)
        max_val = 5000  # max temperature (K)
        opts = 0
        unit = "(K)"
    else:
        min_val = np.amin(y_true[:,:,:,1::5])*1e-9  # min temperature (K)
        max_val = np.amax(y_true[:,:,:,1::5])*1e-9  # max temperature (K)
        unit = "(GPa)"
        opts = 1

    # print(X.shape)
    # plot the prediction results
    x_num = np.linspace(0, 2.38, 14)  # discrete timesteps
    print(x_num)
    fig, ax = plt.subplots(5, 4, figsize=(18, 16))
    if options == "temperature":
        fig.suptitle('Temperature field evolution', fontsize=40)
    else:
        fig.suptitle('Pressure field evolution', fontsize=40)
        
    plt.subplots_adjust(wspace=0.06, hspace= 0.07, top=0.9)
    for i in range(4):
        ax[0][i].clear()
        ax[0][i].clear()
        ax[0][i].set_xticks([])
        ax[0][i].set_yticks([])
        im = ax[0][i].imshow(
            np.squeeze(y_true[case_id, :, :192, ((i) * step + 1)*5 + opts]),
            cmap="jet",
            vmin=min_val,
            vmax=max_val,
        )
        ax[0][i].set_title(("%.2f" % x_num[i * step + 1] + " (ns)"), fontsize=36)
        
        for j in range(4):
            ax[j+1][i].set_xticks([])
            ax[j+1][i].set_yticks([])
            ax[j+1][i].imshow(
                np.squeeze(y_pred[j][case_id, :, :192, (i* step) * 5 + opts]),
                cmap="jet",
                vmin=min_val,
                vmax=max_val,
            )
            
    ax[0][0].set_ylabel("Ground \n truth", fontsize=36)
    ax[1][0].set_ylabel("PARCv2", fontsize=36)
    ax[2][0].set_ylabel("PARC \n (num. int.)", fontsize=36)
    ax[3][0].set_ylabel("PARC \n (NN int.)", fontsize=36)
    ax[4][0].set_ylabel("FNO", fontsize=36)
    fig.subplots_adjust(right=0.95)
    
    cbar_ax = fig.add_axes([0.96, 0.111, 0.025, 0.788])
    norm = mpl.colors.Normalize(vmin=min_val, vmax=max_val)
    cbar = fig.colorbar(
        mpl.cm.ScalarMappable(norm=norm, cmap="jet"), cax=cbar_ax, ax=im
    )
    cbar.set_label(label=unit, weight="bold", fontsize=36)
    cbar.ax.tick_params(labelsize=36)
    fig.savefig('field_evolution_plot_temp_em.png',bbox_inches='tight')
    plt.show()