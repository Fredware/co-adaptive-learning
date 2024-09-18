# This is a sample Python script.

# Press Shift+F10 to execute it or replace it with your code.
# Press Double Shift to search everywhere for classes, files, tool windows, actions, and settings.
import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd
import seaborn as sns

from tqdm import tqdm
from scipy.signal import argrelmin
from sklearn.feature_selection import mutual_info_regression


def create_delayed_hankel(vector, num_lags):
    """
    Creates a truncated Hankel matrix with delayed columns.
    Args:
        vector: The input column vector.
        num_lags: The number of additional columns.
    Returns:
        The truncated Hankel matrix.
    """
    n = len(vector)
    hankel_mat = np.zeros((n - num_lags, num_lags + 1))
    for i in range(n - num_lags):
        hankel_mat[i, :] = vector[i:i + num_lags + 1]
    return hankel_mat


def find_first_local_minimum(data):
    """
    Finds the first local minimum of the data using argrelmin.
    :param data: The input data as a numpy array.
    :return: The index of the first local minimum or None if no local minimum is found.
    """
    minima_idxs = argrelmin(data)[0]
    if minima_idxs.size > 0:
        return minima_idxs[0]
    else:
        return None


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    p_id = "10"
    learning_cond = "cl"

    kdf_filename = f"..\\..\\..\\temp-data\\p_{p_id}-{learning_cond}\\p_{p_id}-{learning_cond}-kdf-mav.csv"
    kdf_mav_idxs_filename = f"..\\..\\..\\temp-data\\p_{p_id}-{learning_cond}\\p_{p_id}-{learning_cond}-kdf-mav-gesture_idxs.csv"
    save_dir = os.path.dirname(kdf_filename)

    kdf_file = pd.read_csv(kdf_filename)
    kdf_mav = kdf_file.filter(like="MAV ")
    kdf_mav_idxs = pd.read_csv(kdf_mav_idxs_filename)

    trial_starts = kdf_mav_idxs["gesture_a_starts_idx"].to_numpy()
    trial_ends = kdf_mav_idxs["gesture_b_stops_idx"].to_numpy()

    num_trials = len(trial_starts)
    num_channels = kdf_mav.shape[1]

    mi_scores_array = np.empty((num_trials, num_channels), dtype=object)
    mi_scores_minima = np.empty((num_trials, num_channels))

    for i in tqdm(range(num_trials)):
        trial_start = trial_starts[i]
        trial_end = trial_ends[i]
        for j in range(num_channels):
            input_vec = kdf_mav.iloc[trial_start:trial_end, j].to_numpy()
            hankel_mat = create_delayed_hankel(input_vec, 300)
            mi_scores = mutual_info_regression(hankel_mat, hankel_mat[:, 0])
            mi_scores_array[i, j] = mi_scores
            mi_scores_minima[i, j] = find_first_local_minimum(mi_scores)
    tau_median = np.median(np.median(mi_scores_minima, axis=0))

    fig, axes = plt.subplots(num_trials, 1, figsize=(8.5, 14))
    for i in range(num_trials):
        for j in range(num_channels):
            axes[i].plot(mi_scores_array[i, j], label=f"Channel {j + 1}")
        axes[i].set_title(f"Trial {i + 1}")
        # axes[i].legend()
        axes[i].grid(True)
    plt.tight_layout()
    plt.show()

    plt.boxplot(mi_scores_minima, vert=True)
    plt.axhline(y=tau_median, color='red', linestyle='-', linewidth=2)
    plt.title(f"Time delay distribution per channel ({p_id}-{learning_cond})")
    plt.xlabel("Channel")
    plt.ylabel(f"Time Delay Distribution (med-med = {tau_median})")
    plt.show()

    np.save(os.path.join(save_dir, f'p_{p_id}-{learning_cond}-mi_scores_minima'), mi_scores_minima)
    np.save(os.path.join(save_dir, f'p_{p_id}-{learning_cond}-mi_scores_array'), mi_scores_array)
