import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pickle
import seaborn as sns

from matplotlib.gridspec import GridSpec
from mpl_toolkits.mplot3d import Axes3D
from sklearn.manifold import MDS
from sklearn.preprocessing import MinMaxScaler
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
    x_labels = [f"T{i+1:02d}" for i in range(len(x_ticks))]
    plt.xticks(x_ticks, x_labels)
    plt.show()


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    p_id = "p_08"
    learning_cond = "ml"
    base_path = f"..\\..\\..\\temp-data\\{p_id}-{learning_cond}\\{p_id}-{learning_cond}"
    trial_mats = np.load(base_path + "-kdf-mav-v.npz")
    kdf_file = pd.read_csv(base_path + "-kdf-mav.csv")
    kdf_mav_idxs = pd.read_csv(base_path + "-kdf-mav-gesture_idxs.csv")
    sims_dmd = np.load(base_path + "-kdf-mav-sim_mat.npy")
    with open(base_path + "-kdf-mav-gesture_labels.pkl", 'rb') as f:
        gesture_labels = pickle.load(f)

    if learning_cond == "ml":
        num_rows = 4
    else:
        num_rows = 3
    num_cols = 3

    gesture_a_starts = kdf_mav_idxs["gesture_a_starts_idx"].to_numpy()
    gesture_a_ends = kdf_mav_idxs["gesture_a_stops_idx"].to_numpy()
    gesture_b_starts = kdf_mav_idxs["gesture_b_starts_idx"].to_numpy()
    gesture_b_ends = kdf_mav_idxs["gesture_b_stops_idx"].to_numpy()

    # Input plot
    fig, axes = plt.subplots(num_rows, num_cols, figsize=(15, 8.5))
    for i in range(len(gesture_a_starts)):
        trial_start = gesture_a_starts[i]
        trial_end = gesture_b_ends[i]
        kdf_trial = kdf_file.iloc[trial_start:trial_end, :]

        row = i // num_cols
        col = i % num_cols
        ax = axes[row, col]
        n = (trial_end - trial_start) // 2

        plot_trial_mav(ax, kdf_trial, n)
        ax.set_title(f"Trial {i + 1}")
    plt.tight_layout()
    plt.show()

    # Variance Plot TBT
    fig, axes = plt.subplots(*(num_rows, 6), figsize=(15, 8.5), subplot_kw={'projection': '3d'})
    fig.tight_layout()
    for i in range(len(gesture_a_starts)):
        trial_start = gesture_a_starts[i]
        trial_end = gesture_b_ends[i]
        kdf_trial = kdf_file.iloc[trial_start:trial_end, :]
        target_cols = [col for col in kdf_trial.columns if col.startswith('Targets_')]
        mav_cols = [col for col in kdf_trial.columns if col.startswith('MAV')]
        trial_data = kdf_trial[target_cols + mav_cols]

        scaler = MinMaxScaler()
        scaled_data = scaler.fit_transform(trial_data[mav_cols])

        if i == 0:
            pca = PCA(n_components=3)
            pca.fit(scaled_data)
        projected_data = pca.transform(scaled_data)
        gesture_split = (trial_end - trial_start) // 2

        ax = axes.flat[2 * i]
        cmap = plt.colormaps['Blues']
        gesture_data = projected_data[:gesture_split, :]
        x, y, z = gesture_data[:, 0], gesture_data[:, 1], gesture_data[:, 2]
        colors = cmap((np.linspace(0.25, 1, len(x))))
        ax.scatter(x, y, z, c=colors, cmap=cmap, s=10)
        ax.set_title(f'Trial {i+1}A')

        ax = axes.flat[2 * i + 1]
        cmap = plt.colormaps['Reds']
        gesture_data = projected_data[gesture_split:, :]
        x, y, z = gesture_data[:, 0], gesture_data[:, 1], gesture_data[:, 2]
        colors = cmap((np.linspace(0.25, 1, len(x))))
        ax.scatter(x, y, z, c=colors, cmap=cmap, s=10)
        ax.set_title(f'Trial {i+1}B')
    plt.show()


    # Variance Plot All
    fig, axes = plt.subplots(*(1, 2), figsize=(15, 8.5), subplot_kw={'projection': '3d'})
    fig.tight_layout()
    for i in range(len(gesture_a_starts)):
        trial_start = gesture_a_starts[i]
        trial_end = gesture_b_ends[i]
        kdf_trial = kdf_file.iloc[trial_start:trial_end, :]
        target_cols = [col for col in kdf_trial.columns if col.startswith('Targets_')]
        mav_cols = [col for col in kdf_trial.columns if col.startswith('MAV')]
        trial_data = kdf_trial[target_cols + mav_cols]

        scaler = MinMaxScaler()
        scaled_data = scaler.fit_transform(trial_data[mav_cols])

        if i == 0:
            pca = PCA(n_components=3)
            pca.fit(scaled_data)
        projected_data = pca.transform(scaled_data)
        gesture_split = (trial_end - trial_start) // 2

        ax = axes.flat[0]
        cmap = plt.colormaps['winter']
        gesture_data = projected_data[:gesture_split, :]
        x, y, z = gesture_data[:, 0], gesture_data[:, 1], gesture_data[:, 2]
        colors = cmap((np.linspace(0.25, 1, 10)))
        ax.scatter(x, y, z, color=colors[i], s=10)
        if i+1 == len(gesture_a_starts):
            ax.set_title(f'Trials 1A-{i+1}A')

        ax = axes.flat[1]
        cmap = plt.colormaps['hot']
        gesture_data = projected_data[gesture_split:, :]
        x, y, z = gesture_data[:, 0], gesture_data[:, 1], gesture_data[:, 2]
        colors = cmap((np.linspace(0.25, 1, 10)))
        ax.scatter(x, y, z, color=colors[i], s=10)
        if i+1 == len(gesture_a_starts):
            ax.set_title(f'Trials 1B-{i+1}B')
    plt.show()

    # Latent Plot
    plot_svd_trajectories(trial_mats, (num_rows, 6))

    # Output plot
    fig, axes = plt.subplots(num_rows, num_cols, figsize=(15, 8.5))
    for i in range(len(gesture_a_starts)):
        trial_start = gesture_a_starts[i]
        trial_end = gesture_b_ends[i]
        kdf_trial = kdf_file.iloc[trial_start:trial_end, :]

        row = i // num_cols
        col = i % num_cols
        ax = axes[row, col]
        n = (trial_end - trial_start) // 2

        plot_trial_inferences(ax, kdf_trial, n)
        ax.set_title(f"Trial {i + 1}")
    plt.tight_layout()
    plt.show()

    # Performance Plot
    gesture_a_trmse = compute_gesture_trmse(kdf_file, gesture_a_starts, gesture_a_ends)
    gesture_b_trmse = compute_gesture_trmse(kdf_file, gesture_b_starts, gesture_b_ends)
    plot_trial_trmse(gesture_a_trmse, gesture_b_trmse)

    # Similarity Plot
    sns.heatmap(sims_dmd, cmap="BuPu")
    plt.show()
    sns.heatmap(sims_dmd[::2, ::2], cmap='Blues')
    plt.show()
    sns.heatmap(sims_dmd[1::2, 1::2], cmap='Reds')
    plt.show()

    df = pd.DataFrame()
    df['Model Type'] = gesture_labels
    reduced = MDS(dissimilarity='precomputed').fit_transform(sims_dmd)
    df["0"] = reduced[:, 0]
    df["1"] = reduced[:, 1]

    palette = 'BuPu'
    sns.scatterplot(data=df, x="0", y="1", hue="Model Type", palette=palette)
    plt.xlabel(f"MDS 1")
    plt.ylabel(f"MDS 2")
    plt.tight_layout()
    plt.show()
