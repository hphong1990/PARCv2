import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
import tensorflow as tf
from scipy.ndimage import gaussian_filter
from PIL import ImageFilter, Image
from IPython.display import display
from matplotlib.patches import Rectangle


def plot_field_evolution(y_pred, y_true, case_id):
    """Visualize the prediction
    :param y_pred:          (numpy) predicted fields
                            [0, ...] predicted fields (F)
                            [1, ...] predicted change of fields (F_dot)
    :param y_true:          (numpy) true label of the fields
    :param test_sample_no:  (int)   array index to select the test case
    :state_var_type:        (str)   indicate which fields to plot the result to apply correct scaling
    """

    step = 17
    max_val = 2.5  
    min_val = 0  
    unit = "(m/s)"
    Y, X = np.mgrid[0:127:128j, 0:255:256j]
    print(X.shape)
    # plot the prediction results
    x_num = np.linspace(0, 2, 38)  # discrete timesteps
    print(x_num)
    fig, ax = plt.subplots(5,3, figsize=(20, 20))
    fig.suptitle('Velocity field evolution: $u = \sqrt{u_x^2 + u_y^2}$\nRe = 550', fontsize=44)
    plt.subplots_adjust(wspace=0.06, hspace= 0.06, top=0.85)
    for i in range(3):
        ax[0][i].clear()
        ax[0][i].clear()
        ax[0][i].set_xticks([])
        ax[0][i].set_yticks([])
        
        # Plot ground truth
        im = ax[0][i].imshow(
            np.squeeze(np.sqrt(y_true[case_id, :, :, (i * step + 3) * 3 + 0]**2 + y_true[case_id, :, :, (i * step + 3) * 3 + 1]**2)),
            # cmap="bwr",
            vmin=0,
            vmax=max_val,
        )
        ax[0][i].set_title(("%.2f" % x_num[i * step + 3] + " (s)"), fontsize = 44)

        # Plot prediction
        for j in range(4):
            ax[j+1][i].imshow(
                np.squeeze(np.sqrt(y_pred[j][case_id, :, :, (i * step + 3) * 3 + 0]**2 + y_pred[j][case_id, :, :, (i * step + 3) * 3 + 1]**2)),
                # cmap="bwr",
                vmin=0,
                vmax=max_val,
            )
            ax[j+1][i].set_xticks([])
            ax[j+1][i].set_yticks([])
    ax[0][0].set_ylabel("Ground \n truth", fontsize=40)
    ax[1][0].set_ylabel("PARCv2", fontsize=40)
    ax[2][0].set_ylabel("FNO", fontsize=40)
    ax[3][0].set_ylabel("PhyCRNet", fontsize=40)
    ax[4][0].set_ylabel("PI-FNO", fontsize=40)
    fig.subplots_adjust(right=0.95)
    
    cbar_ax = fig.add_axes([0.96, 0.114, 0.025, 0.733])
    norm = mpl.colors.Normalize(vmin=min_val, vmax=max_val)
    cbar = fig.colorbar(
        mpl.cm.ScalarMappable(norm=norm, cmap="viridis"), cax=cbar_ax, ax=im
    )
    cbar.set_label(label=unit, weight="bold", fontsize=44)
    cbar.ax.tick_params(labelsize=44)

    fig.savefig('field_evolution_plot_ns.png',bbox_inches='tight')
    plt.show()

def pde_mse_div_Re(mse_whole, pde_whole, div_whole, Re_list):
    # MSE/PDE vs R
    name = ['PARCv2','FNO','PhyCRNet','PIFNO']
    linestyle=['-', '-', '-', '-', '--', '--']
    color = ['blue','green','black','red']

    fig, ax = plt.subplots(1, 3, figsize=(19, 5), gridspec_kw={"hspace":0.5, "wspace": 0.6})
    plt.subplots_adjust(top=0.8) 

    # MSE vs R
    for i in range(4):
        ax[0].plot(Re_list, mse_whole[i], marker = 'o', color = color[i], linewidth = 1.5, markersize = 6, linestyle = linestyle[i])

    ax[0].add_patch(Rectangle((100, 0), 800, 100, fc = 'gray', ec = 'none', alpha = 0.7))
    ax[0].set_ylabel("RMSE $(m/s)$", fontsize=22)
    ax[0].set_xlabel("Re", fontsize=22)

    # PDE vs R
    for i in range(4):
        ax[1].plot(Re_list, pde_whole[i], "o", color = color[i], linewidth = 1.5, markersize = 6, linestyle = linestyle[i])
        
    ax[1].set_ylabel(r"PDE residual $(m/s^2)$", fontsize=22)
    ax[1].set_xlabel("Re", fontsize=22)
    ax[1].add_patch(Rectangle((100, 0), 800, 10000, fc = 'gray', ec = 'none', alpha = 0.7))

    # Div. cond vs R
    for i in range(4):
        ax[2].plot(Re_list, div_whole[i], "o", color = color[i], linewidth = 1.5, markersize = 6, linestyle = linestyle[i], label=name[i])
        
    ax[2].set_ylabel("Divergent-free condition \n error $(1/s)$", fontsize=22)
    ax[2].set_xlabel("Re", fontsize=22)
    ax[2].add_patch(Rectangle((100, 0), 800, 20, fc = 'gray', ec = 'none', alpha = 0.7, label = 'Covered by training set'))

    ax[2].set_yscale("log")
    ax[2].tick_params(axis='both', which='major', labelsize=20)

    ax[2].set_ylim(0.001,15)

    # Denote the training R range
    for i in range(2):
        ax[i].set_yscale("log")
        ax[i].tick_params(axis='both', which='major', labelsize=20)
    ax[0].set_ylim(0.01,0.5)
    ax[1].set_ylim(0.5,5)
    fig.legend(fontsize=22, ncol = 5, loc = 8,  bbox_to_anchor=(0.51, -0.18))

    fig.savefig("./mse_pde_R_ns.png", bbox_inches='tight', pad_inches=0.05)



