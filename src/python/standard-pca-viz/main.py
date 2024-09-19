import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pickle
import seaborn as sns

from matplotlib.gridspec import GridSpec
from mpl_toolkits.mplot3d import Axes3D
from sklearn.manifold import MDS
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.decomposition import PCA


def plot_trial_mav(ax, kdf_trial, n):
    """
    Normalizes each channel of MAV, plots the channel data against the targets, and color codes the data before and
    after n.
    :param ax: Matplotlib axes object
    :param kdf_trial: DataFrame containing KDF trial data
    :param n: Number of data points to plot in blue from the beginning
    :return: None
    """
    target_cols = [col for col in kdf_trial.columns if col.startswith('Targets_')]
    mav_cols = [col for col in kdf_trial.columns if col.startswith('MAV')]
    data = kdf_trial[target_cols + mav_cols]

    scaler = MinMaxScaler()
    data[mav_cols] = scaler.fit_transform(data[mav_cols])

    sns_blue = sns.color_palette("Paired")[1]  # '#115BA4'
    sns_red = sns.color_palette("Paired")[5]

    for col in mav_cols:
        ax.plot(data.index[:n], data[col][:n], color=sns_blue, alpha=0.1)
        ax.plot(data.index[n:], data[col][n:], color=sns_red, alpha=0.1)

    for i, col in enumerate(target_cols):
        if i < len(target_cols) - 2:
            l_style = '-.'
        else:
            l_style = '-'
        l_width = 4
        ax.plot(data.index[:n], data[col][:n], color=sns_blue, alpha=0.2, linestyle=l_style, linewidth=l_width)
        ax.plot(data.index[n:], data[col][n:], color=sns_red, alpha=0.2, linestyle=l_style, linewidth=l_width)
    #
    # for col in target_cols:
    #     ax.plot(data.index[:n], data[col][:n], color=sns_blue, alpha=0.5)
    #     ax.plot(data.index[n:], data[col][n:], color=sns_red, alpha=0.5)

    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.set_xticks([])
    ax.set_yticks([0, 0.5, 1])


def plot_svd_trajectories(trial_mats, grid_size):
    # Create list from dict
    trial_mat_list = [trial_mats[f'arr_{i}'] for i in range(len(trial_mats))]

    # Validate grid size
    num_trials = len(trial_mat_list)
    if num_trials > grid_size[0] * grid_size[1]:
        raise ValueError('grid size does not match')

    # Configure plot
    fig, axes = plt.subplots(*grid_size, figsize=(15, 8.5), subplot_kw={'projection': '3d'})
    fig.tight_layout()

    trial_num = 1
    trial_letter = 'A'

    for i, (ax, trial) in enumerate(zip(axes.flat, trial_mat_list)):
        x, y, z = trial[:, 0], trial[:, 1], trial[:, 2]
        if i % 2 == 0:
            cmap = plt.colormaps['Blues']
        else:
            cmap = plt.colormaps['Reds']
        colors = cmap((np.linspace(0.25, 1, len(x))))
        ax.scatter(x, y, z, c=colors, cmap=cmap, s=10)
        ax.set_xticklabels([])
        ax.set_yticklabels([])
        ax.set_zticklabels([])
        ax.set_title(f'Trial {trial_num}{trial_letter}')
        if (i + 1) % 2 == 0:
            trial_num += 1
            trial_letter = 'A'
        else:
            trial_letter = 'B'
    plt.show()


def plot_trial_inferences(ax, kdf_trial, n):
    """
    Plots the Targets against the Kalman output for the trial data. It also plots the first n data points with blue.
    :param ax: Axes object
    :param kdf_trial: DataFrame containing KDF trial data
    :param n: Number of data points to plot in blue from the beginning
    :return: None
    """
    target_cols = [col for col in kdf_trial.columns if col.startswith('Targets_')]
    kalman_cols = [col for col in kdf_trial.columns if col.startswith('Kalman')]
    data = kdf_trial[target_cols + kalman_cols]

    sns_blue = sns.color_palette("Paired")[1]
    sns_red = sns.color_palette("Paired")[5]

    for i, col in enumerate(kalman_cols):
        if i < len(kalman_cols) - 2:
            l_style = '--'
            sns_blue = "#06068b"
            sns_red = "#971115"
        else:
            l_style = '-'
            sns_blue = "#0a0af1"
            sns_red = "#fd1c23"
        l_width = 2.5
        ax.plot(data.index[:n], data[col][:n], color=sns_blue, alpha=0.5, linestyle=l_style, lw=l_width)
        ax.plot(data.index[n:], data[col][n:], color=sns_red, alpha=0.5, linestyle=l_style, lw=l_width)

    for i, col in enumerate(target_cols):
        if i < len(target_cols) - 2:
            l_style = '--'
            sns_blue = "#000000"
            sns_red = "#000000"
            l_width = 5.5
        else:
            l_style = '-'
            sns_blue = "#5e5e5e"
            sns_red = "#5e5e5e"
            l_width = 5
        ax.plot(data.index[:n], data[col][:n], color=sns_blue, alpha=0.5, linestyle=l_style, linewidth=l_width)
        ax.plot(data.index[n:], data[col][n:], color=sns_red, alpha=0.5, linestyle=l_style, linewidth=l_width)

    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.set_xticks([])
    ax.set_yticks([0, 0.5, 1])


def compute_gesture_trmse(kdf_file, gesture_starts, gesture_ends):
    """
    Compute the hard-thresholded RMSE of the gestures specified by gestures_starts and gesture_ends.
    :param kdf_file: KDF file of the session
    :param gesture_starts: list of indices of gestures starts
    :param gesture_ends: list of indices of gestures ends
    :return: List of hard-thresholded RMSEs
    """
    hard_tresh = 0.10
    gesture_trmse = []
    for i in range(len(gesture_starts)):
        gesture_start = gesture_starts[i]
        gesture_end = gesture_ends[i]
        kdf_gesture = kdf_file.iloc[gesture_start:gesture_end, :]
        target_cols = [col for col in kdf_gesture.columns if col.startswith('Targets_')]
        kalman_cols = [col for col in kdf_gesture.columns if col.startswith('Kalman')]
        targets = kdf_gesture[target_cols]
        estimates = kdf_gesture[kalman_cols]
        estimates.columns = targets.columns
        errors = targets.subtract(estimates)
        rms_errors = np.sqrt(errors.pow(2).mean())
        thresh_errors = rms_errors.copy()
        thresh_errors[thresh_errors <= hard_tresh] = 0
        trmse_d13 = thresh_errors[:3]
        trmse_d45 = thresh_errors[3:]
        gesture_trmse.append(np.mean([trmse_d13.mean(), trmse_d45.mean()]))
    return gesture_trmse


def plot_trial_trmse(gesture_a_trmse, gesture_b_trmse):
    """
    Plots two TRMSE sequences by interleaving them and painting the first one blue.
    :param gesture_a_trmse: Co-activation sequence (blue)
    :param gesture_b_trmse: Differential activation sequence (red)
    :return: None
    """
    # Combine the lists into a single list
    combined_gestures_trmse = gesture_a_trmse + gesture_b_trmse

    # Create x-axis values
    x = list(range(len(combined_gestures_trmse)))

    # Create y-axis values
    y = combined_gestures_trmse

    # Plot the lines
    plt.plot(x[::2], gesture_a_trmse, 'o-', color=sns.color_palette("Paired")[1], label='Co-Activation')
    plt.plot(x[1::2], gesture_b_trmse, 'v-', color=sns.color_palette("Paired")[5], label='Differential Activation')

    # Add labels and legend
    plt.xlabel('Trial')
    plt.ylabel('TRMSE')
    plt.legend()
    plt.ylim(0, 0.75)
    ax = plt.gca()
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    x_ticks = np.arange(0.5, 18.5, 2)
    x_labels = [f"T{i + 1:02d}" for i in range(len(x_ticks))]
    plt.xticks(x_ticks, x_labels)
    plt.show()


def load_dataset(p_id: str, learning_cond: str) -> dict:
    """
    Load the kdf files and the associated event idxs stored as starts and stops for each gesture
    :param p_id: Participant ID
    :param learning_cond: Learning condition (ml, cl, or hl)
    :return:
    """
    base_path = f"..\\..\\..\\temp-data\\{p_id}-{learning_cond}\\{p_id}-{learning_cond}"
    kdf_file = pd.read_csv(base_path + "-kdf-mav.csv")
    kdf_mav_idxs = pd.read_csv(base_path + "-kdf-mav-gesture_idxs.csv")
    dataset = dict([("kdf", kdf_file), ("idxs", kdf_mav_idxs)])
    return dataset


def get_nth_trial_mav(dataset: dict, n: int) -> np.ndarray:
    """
    Takes in a dataset and returns the nth trial.
    :param dataset: dictionary containing kdf data and corresponding event idxs
    :param n: trial index starting from 0
    :return: mav data as a numpy array of N x Channels
    """
    idxs = dataset["idxs"]
    gesture_a_starts = idxs["gesture_a_starts_idx"].to_numpy()
    gesture_a_ends = idxs["gesture_a_stops_idx"].to_numpy()
    gesture_b_starts = idxs["gesture_b_starts_idx"].to_numpy()
    gesture_b_ends = idxs["gesture_b_stops_idx"].to_numpy()

    kdf = dataset["kdf"]
    trial_start = gesture_a_starts[n]
    trial_end = gesture_b_ends[n]
    kdf_trial = kdf.iloc[trial_start:trial_end, :]
    mav_cols = [col for col in kdf_trial.columns if col.startswith('MAV')]
    mav_trial = kdf_trial[mav_cols]
    print(mav_trial.shape)
    return mav_trial


def plot_pca(ax, data, n_dim, data_color):
    """
    Plots data as a scatter plot in the specified axis.
    :param ax: Axis to plot
    :param data: 2D or 3D data
    :param n_dim: number of columns in data
    :param data_color: color of data
    :return: None
    """
    dot_size = 50
    dot_alpha = 0.25
    if n_dim == 3:
        x, y, z = data[:, 0], data[:, 1], data[:, 2]
        ax.scatter(x, y, z, color=data_color, s=dot_size, alpha=dot_alpha)
    elif n_dim == 2:
        x, y = data[:, 0], data[:, 1]
        ax.scatter(x, y, color=data_color, s=dot_size, alpha=dot_alpha)
    else:
        raise ValueError("n_dim must be 2 or 3")


def get_pca_projection(pca_ndim, learning_conds, p_id, num_trials=1):
    mav_all_conds = np.empty((0, 32))
    for cond in learning_conds:
        cond_dataset = load_dataset(p_id, cond)
        for trial in range(num_trials):
            if cond == "ml":
                # Skip the zeroth trial of ML
                mav_cond = get_nth_trial_mav(cond_dataset, trial + 1)
            else:
                mav_cond = get_nth_trial_mav(cond_dataset, trial)
            norm_mav = StandardScaler().fit_transform(MinMaxScaler().fit_transform(mav_cond))
            mav_all_conds = np.vstack((mav_all_conds, norm_mav))
    pca_anchor = PCA(n_components=pca_ndim)
    pca_anchor.fit(mav_all_conds)
    print(f'Axis variances: {pca_anchor.explained_variance_ratio_}')
    print(f'Total variance: {np.sum(pca_anchor.explained_variance_ratio_)}')
    return pca_anchor


def get_zeroth_pca_projection(pca_ndim, p_id):
    cond_dataset = load_dataset(p_id, 'ml')
    mav_cond = get_nth_trial_mav(cond_dataset, 0)
    norm_mav = StandardScaler().fit_transform(MinMaxScaler().fit_transform(mav_cond))
    pca_anchor = PCA(n_components=pca_ndim)
    pca_anchor.fit(norm_mav)
    print(f'Axis variances: {pca_anchor.explained_variance_ratio_}')
    print(f'Total variance: {np.sum(pca_anchor.explained_variance_ratio_)}')
    return pca_anchor


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    p_id = "p_10"
    learning_conds = ["ml", "cl", "hl"]

    pca_ndim = 2
    # pca_anchor = get_pca_projection(pca_ndim, learning_conds, p_id, 9)
    pca_anchor = get_zeroth_pca_projection(pca_ndim, p_id)

    ax_mins, ax_maxs = [float('inf')] * pca_ndim, [float('-inf')] * pca_ndim
    for cond in learning_conds:
        cond_dataset = load_dataset(p_id, cond)
        kdf = cond_dataset["kdf"]
        mav_cols = [col for col in kdf.columns if col.startswith('MAV')]
        cond_mav = kdf[mav_cols]
        norm_mav = StandardScaler().fit_transform(MinMaxScaler().fit_transform(cond_mav))
        projected_dataset = pca_anchor.transform(norm_mav)
        ax_mins = [min(a, b) for a, b in zip(ax_mins, projected_dataset.mean(axis=0)-4*projected_dataset.std(axis=0))]
        ax_maxs = [max(a, b) for a, b in zip(ax_maxs, projected_dataset.mean(axis=0)+4*projected_dataset.std(axis=0))]

    colors = sns.color_palette("hls", as_cmap=False, n_colors=10)
    for cond in learning_conds:
        sns.set()
        if pca_ndim == 3:
            fig, axes = plt.subplots(*(1, 2), figsize=(15, 8.5), subplot_kw={'projection': '3d'})
            xmin, ymin, zmin = ax_mins
            xmax, ymax, zmax = ax_maxs
        elif pca_ndim == 2:
            fig, axes = plt.subplots(*(1, 2), figsize=(15, 8.5))
            xmin, ymin = ax_mins
            xmax, ymax = ax_maxs
        fig.tight_layout()

        for ax in axes.flat:
            ax.set_xlim(xmin, xmax)
            ax.set_ylim(ymin, ymax)
            if pca_ndim == 3:
                ax.set_zlim(zmin, zmax)

        cond_dataset = load_dataset(p_id, cond)
        for trial in range(9):
            if cond == "ml":
                mav_trial = get_nth_trial_mav(cond_dataset, trial + 1)
            else:
                mav_trial = get_nth_trial_mav(cond_dataset, trial)
            norm_mav = StandardScaler().fit_transform(MinMaxScaler().fit_transform(mav_trial))
            projected_mav = pca_anchor.transform(norm_mav)
            # TODO: write function to split projection based on gestured indices
            # Split rows evenly into two arrays
            gesture_a, gesture_b = np.array_split(projected_mav, 2)
            plot_pca(axes.flat[0], gesture_a, pca_ndim, colors[trial])
            plot_pca(axes.flat[1], gesture_b, pca_ndim, colors[trial])
        fig.suptitle(f'{p_id.upper()} - {cond.upper()}', fontsize=30)
        plt.show()

    colors = sns.color_palette("hls", as_cmap=False, n_colors=10)
    for cond in learning_conds:
        sns.set()
        if pca_ndim == 3:
            fig, axes = plt.subplots(*(1, 2), figsize=(15, 8.5), subplot_kw={'projection': '3d'})
            xmin, ymin, zmin = ax_mins
            xmax, ymax, zmax = ax_maxs
        elif pca_ndim == 2:
            fig, axes = plt.subplots(*(1, 2), figsize=(15, 8.5))
            xmin, ymin = ax_mins
            xmax, ymax = ax_maxs
        fig.tight_layout()

        for ax in axes.flat:
            ax.set_xlim(xmin, xmax)
            ax.set_ylim(ymin, ymax)
            if pca_ndim == 3:
                ax.set_zlim(zmin, zmax)

        cond_dataset = load_dataset(p_id, cond)
        for trial, ax in zip([1, 9], axes.flat):
            if cond == "ml":
                mav_trial = get_nth_trial_mav(cond_dataset, trial)
            else:
                mav_trial = get_nth_trial_mav(cond_dataset, trial-1)
            norm_mav = StandardScaler().fit_transform(MinMaxScaler().fit_transform(mav_trial))
            projected_mav = pca_anchor.transform(norm_mav)
            # TODO: write function to split projection based on gestured indices
            # Split rows evenly into two arrays
            gesture_a, gesture_b = np.array_split(projected_mav, 2)
            plot_pca(ax, gesture_a, pca_ndim, "#084489")
            plot_pca(ax, gesture_b, pca_ndim, "#B21218")
        fig.suptitle(f'{p_id.upper()} - {cond.upper()}', fontsize=30)
        plt.show()
