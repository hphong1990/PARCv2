import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
from matplotlib.patches import Rectangle

# import tensorflow as tf
# from scipy.ndimage import gaussian_filter
# from PIL import ImageFilter, Image
# from IPython.display import display

def plot_field_evolution(y_pred, y_true, case_id):
    """Visualize the prediction
    :param y_pred:          (list of numpy) predicted fields
    :param y_true:          (numpy) true label of the fields
    :case_id:               (int)   the index of the case that will be plotted
    :state_var_type:        (str)   indicate which fields to plot the result to apply correct scaling
    """

    # get correct scaling terms
    step = 24
    min_val = 0  # min value 
    max_val = 1  # max value 
    unit = "(cm/s)"
    Y, X = np.mgrid[0:63:64j, 0:63:64j]

    # Create the time label
    x_num = np.linspace(0, 2, 101)  # discrete timesteps
    
    fig, ax = plt.subplots(5, 5, figsize=(15,16.5))
    fig.suptitle('Velocity field evolution: $u = \sqrt{u_x^2 + u_y^2}$\nRe = 3000, a = 0.75, w = 0.85', fontsize=30)
    plt.subplots_adjust(wspace=0.06, hspace=-0.07, top=0.88)

    # Loop throught the time
    for i in range(5):
        ax[0][i].clear()
        ax[0][i].clear()
        ax[0][i].set_xticks([])
        ax[0][i].set_yticks([])

        # Plot ground truth
        im = ax[0][i].imshow(
            np.squeeze(np.sqrt(y_true[case_id, :, :, (i * step+3) * 3 + 0]**2 + y_true[case_id, :, :, (i * step+3) * 3 + 1]**2)),
            # cmap="jet",
            vmin=0,
            vmax=1,
        )
        
        ax[0][i].set_title(("%.2f" % x_num[i * step+3] + " (s)"), fontsize=30)

        # Plot the prediction
        for j in range(4):
            ax[j+1][i].set_xticks([])
            ax[j+1][i].set_yticks([])
            ax[j+1][i].imshow(
                np.squeeze(np.sqrt(y_pred[j][case_id, :, :, (i * step+3) * 3 + 0]**2 + y_pred[j][case_id, :, :, (i * step+3) * 3 + 1]**2)),
                # cmap="jet",
                vmin=0,
                vmax=1,
            )
            
    ax[0][0].set_ylabel("Ground \n truth", fontsize=30)
    ax[1][0].set_ylabel("PARCv2", fontsize=30)
    ax[2][0].set_ylabel("FNO", fontsize=30)
    ax[3][0].set_ylabel("PhyCRNet", fontsize=30)
    ax[4][0].set_ylabel("PI-FNO", fontsize=30)
    fig.subplots_adjust(right=0.95)
    
    cbar_ax = fig.add_axes([0.96, 0.119, 0.025, 0.752])
    norm = mpl.colors.Normalize(vmin=min_val, vmax=max_val)
    cbar = fig.colorbar(
        mpl.cm.ScalarMappable(norm=norm, cmap="viridis"), cax=cbar_ax, ax=im
    )
    cbar.set_label(label=unit, weight="bold", fontsize=26)
    cbar.ax.tick_params(labelsize=26)
    fig.savefig('field_evolution_plot_burgers.png',bbox_inches='tight')
    plt.show()

def pde_mse_R(mse_whole, pde_whole, R_list):

    # MSE/PDE vs R
    name = ['PARCv2', 'FNO','PhyCRNet','PIFNO']
    linestyle=['-', '-', '-','-', '--', '--']
    color = ['blue','green','black','red']

    fig, ax = plt.subplots(1, 2, figsize=(12, 5), gridspec_kw={"hspace":0.5, "wspace": 0.5})
    plt.subplots_adjust(top=0.8) 

    # Plot MSE vs. R
    for i in range(4):
        mse_pinn_r = np.nanmean(mse_whole[i], axis=(1,2))
        ax[0].plot(R_list, mse_pinn_r, marker = 'o', color = color[i], linewidth = 1.5, markersize = 6, linestyle = linestyle[i], label=name[i])

    ax[0].add_patch(Rectangle((1000, 0), 9000, 1, fc = 'gray', ec = 'none', alpha = 0.7, label = 'Covered by training set'))
    ax[0].set_xscale("log")
    ax[0].set_ylabel("RMSE $(cm/s)$", fontsize=22)
    ax[0].set_xlabel("Re", fontsize=22)

    # Plot PDE residual vs. R
    for i in range(4):
        pde_pinn_r = np.mean(pde_whole[i], axis=(1,2))
        ax[1].plot(R_list, pde_pinn_r, "o", linewidth = 1.5, color = color[i], markersize = 6, linestyle = linestyle[i])
    
    # Some plot setting
    ax[1].set_xscale("log")
    ax[1].set_ylabel(r"PDE residual $(cm/s^2)$", fontsize=22)
    ax[1].set_xlabel("Re", fontsize=22)
    ax[1].add_patch(Rectangle((1000, 0), 9000, 1, fc = 'gray', ec = 'none', alpha = 0.7))

    for i in range(2):
        ax[i].set_yscale("log")
        ax[i].tick_params(axis='both', which='major', labelsize=20)
    ax[0].set_ylim(0.0005,0.1)
    ax[1].set_ylim(0.02,0.2)
    ax[1].tick_params(axis='y', which='minor', labelsize=18)

    fig.legend(fontsize=22, ncol = 2, loc = 8,  bbox_to_anchor=(0.51, -0.38))

    fig.savefig("./mse_pde_R.png", bbox_inches='tight', pad_inches=0.05)