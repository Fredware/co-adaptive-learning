# This is a sample Python script.

# Press Shift+F10 to execute it or replace it with your code.
# Press Double Shift to search everywhere for classes, files, tool windows, actions, and settings.

import seaborn as sns
import pickle
import pandas as pd
import os
import numpy as np
import matplotlib.pyplot as plt
import json

from sklearn.manifold import MDS
from DSA.dmd import DMD
from DSA.simdist import SimilarityTransformDist


def make_labels_from_list(original_list, constant_string):
    """
    Creates a new list where each element is a string combining the index of the corresponding element in the
    original list and constant_string
    :param original_list: The original list.
    :param constant_string: The constant string to append to the index.
    :return: A new list of strings.
    """
    labeled_list = []
    for index, _ in enumerate(original_list):
        label = f"T{index + 1}D{constant_string}"
        labeled_list.append(label)
    return labeled_list


def interleave_lists(list1, list2):
    """
    Interleaves two lists of the same length.
    :param list1:
    :param list2:
    :return: interleaved list
    """
    assert len(list1) == len(list2)

    combined_list = []
    for i in range(len(list1)):
        combined_list.extend([list1[i], list2[i]])
    return combined_list


class Config:
    def __init__(self, config_file):
        with open(config_file, 'r') as f:
            config = json.load(f)
        self.participant_id = config['participant_id']
        self.learning_condition = config['learning_condition']
        self.source_file = config['source_file']
        self.feature_type = config['feature_type']
        self.save_dir_base = config['save_dir_base']
        self.save_dir_format = config['save_dir_format']
        self.save_dir = os.path.join(self.save_dir_base, self.save_dir_format.format(
            participant_id=self.participant_id,
            learning_condition=self.learning_condition))


def generate_filename(config, file_extension, file_identifier=""):
    """
    Generates a filename based on the config object
    :param config: From params.json
    :param file_identifier: name of the variable to save
    :param file_extension: .csv, .npy, .npz
    :return: Generated filename as a string
    """
    components = [config.participant_id, config.learning_condition, config.source_file, config.feature_type]
    if file_identifier != "":
        components.append(file_identifier)
    return '-'.join(components) + file_extension


if __name__ == '__main__':
    config = Config("params.json")
    tau = 7
    m = 8

    kdf_filename = os.path.join(config.save_dir, generate_filename(config,".csv"))
    kdf_file = pd.read_csv(kdf_filename)
    kdf_mav = kdf_file.filter(like="MAV ")
    idxs_filename = os.path.join(config.save_dir, generate_filename(config,".csv", "gesture_idxs"))
    kdf_mav_idxs = pd.read_csv(idxs_filename)

    gesture_a_starts = kdf_mav_idxs["gesture_a_starts_idx"].to_numpy()
    gesture_a_ends = kdf_mav_idxs["gesture_a_stops_idx"].to_numpy()
    gesture_b_starts = kdf_mav_idxs["gesture_b_starts_idx"].to_numpy()
    gesture_b_ends = kdf_mav_idxs["gesture_b_stops_idx"].to_numpy()
    gesture_a_labels = make_labels_from_list(gesture_a_starts, constant_string='A')
    gesture_b_labels = make_labels_from_list(gesture_b_starts, constant_string='B')

    gesture_starts = interleave_lists(gesture_a_starts, gesture_b_starts)
    gesture_ends = interleave_lists(gesture_a_ends, gesture_b_ends)
    gesture_labels = interleave_lists(gesture_a_labels, gesture_b_labels)

    dmd_mat_list = []
    v_mat_list = []
    for i, gesture_start in enumerate(gesture_starts):
        gesture_end = gesture_ends[i]
        df_gesture = kdf_mav.iloc[gesture_start:gesture_end, :]
        gesture_emg_sigs = df_gesture.to_numpy()
        gesture_dmd = DMD(gesture_emg_sigs, n_delays=tau, delay_interval=m)
        gesture_dmd.fit()
        dmd_mat_list.append(gesture_dmd.A_v.numpy())
        v_mat_list.append(gesture_dmd.V)

    print("Finished DMD")

    dmd_tot = len(dmd_mat_list)

    sims_dmd = np.zeros((dmd_tot, dmd_tot))
    sims_type = np.zeros((dmd_tot, dmd_tot))
    comparison_dmd = SimilarityTransformDist(iters=2000, lr=1e-3, score_method="wasserstein", wasserstein_compare='sv')

    for i, mi in enumerate(dmd_mat_list):
        print(i)
        for j, mj in enumerate(dmd_mat_list):
            stype = int(gesture_labels[i] == gesture_labels[j])
            sims_type[i, j] = sims_type[j, i] = stype
            if i == j:
                sims_type[i, i] = 2
            if j < i:
                continue
            sdmd = comparison_dmd.fit_score(mi, mj, zero_pad=True)
            print(i, j, sdmd)

            sims_dmd[i, j] = sims_dmd[j, i] = sdmd

    print("Finished Similarity Calculation")

    sns.heatmap(sims_dmd)
    plt.show()

    df = pd.DataFrame()
    df['Model Type'] = gesture_labels
    reduced = MDS(dissimilarity='precomputed').fit_transform(sims_dmd)
    df["0"] = reduced[:, 0]
    df["1"] = reduced[:, 1]

    palette = 'plasma'
    sns.scatterplot(data=df, x="0", y="1", hue="Model Type", palette=palette)
    plt.xlabel(f"MDS 1")
    plt.ylabel(f"MDS 2")
    plt.tight_layout()
    plt.show()

    np.savez(os.path.join(config.save_dir, generate_filename(config, ".npz", "dmd")), *dmd_mat_list)
    np.savez(os.path.join(config.save_dir, generate_filename(config, ".npz", "v")), *v_mat_list)
    np.save(os.path.join(config.save_dir, generate_filename(config, ".npy", "sim_mat")), sims_dmd)
    with open(os.path.join(config.save_dir, generate_filename(config, ".pkl", "gesture_labels")), 'wb') as f:
        pickle.dump(gesture_labels, f)
