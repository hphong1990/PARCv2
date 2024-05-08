import tensorflow as tf
from scipy.ndimage import gaussian_filter
from PIL import ImageFilter, Image
from IPython.display import display 
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl

def plot_field_evolution(y_pred, y_true, case_id, options, slice_height):
    """Visualize the prediction
    :param y_pred:       (list of numpy arrays) predicted fields
    :param y_true:       (numpy array) true label of the fields
    :param case_id:      (int)   array index to select the test case
    :param options:      (str)   indicate which fields to plot the result to apply correct scaling
    :param slice_height: (int)   height of the image slice to consider
    """
    # Constants
    step = 4
    num_pred = len(y_pred)
    titles = ["Ground\ntruth"] + [f"Prediction {i+1}" for i in range(num_pred)]
    unit = "(K)" if options == "temperature" else "(GPa)"
    field_range = (300, 5000) if options == "temperature" else (np.amin(y_true[:,:,:,1::5])*1e-9, np.amax(y_true[:,:,:,1::5])*1e-9)

    # Determine figure size based on number of predictions
    fig_height = 4 + num_pred * 4
    fig, ax = plt.subplots(num_pred + 1, 4, figsize=(18, fig_height))
    fig.suptitle('Temperature field evolution' if options == "temperature" else 'Pressure field evolution', fontsize=40)
    plt.subplots_adjust(wspace=0.06, hspace= 0.07, top=0.9)

    for i in range(4):
        for j in range(num_pred + 1):
            ax[j][i].clear()
            ax[j][i].set_xticks([])
            ax[j][i].set_yticks([])
            cmap = ax[j][i].imshow(
                np.squeeze(y_true[case_id, :, :slice_height, (i * step + 1)*5 + (0 if options == "temperature" else 1)]),
                cmap="jet",
                vmin=field_range[0],
                vmax=field_range[1],
            )
            ax[j][i].set_title("%.2f (ns)" % (i * step + 1), fontsize=36) if j == 0 else None
            if j > 0:
                cmap = ax[j][i].imshow(
                    np.squeeze(y_pred[j-1][case_id, :, :slice_height, (i* step) * 5 + (0 if options == "temperature" else 1)]),
                    cmap="jet",
                    vmin=field_range[0],
                    vmax=field_range[1],
                )

    for i in range(num_pred + 1):
        ax[i][0].set_ylabel(titles[i], fontsize=36)

    fig.subplots_adjust(right=0.95)
    cbar_ax = fig.add_axes([0.96, 0.111, 0.025, 0.788])
    norm = mpl.colors.Normalize(vmin=field_range[0], vmax=field_range[1])
    cbar = fig.colorbar(mpl.cm.ScalarMappable(norm=norm, cmap="jet"), cax=cbar_ax)
    cbar.set_label(label=unit, weight="bold", fontsize=36)
    cbar.ax.tick_params(labelsize=36)
    plt.show()

    # Save the figure
    fig.savefig('field_evolution_plot_temp_em.png', bbox_inches='tight')


