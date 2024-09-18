# This is a sample Python script.

# Press Shift+F10 to execute it or replace it with your code.
# Press Double Shift to search everywhere for classes, files, tool windows, actions, and settings.
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from DSA import SimilarityTransformDist
from sklearn.manifold import MDS

def print_hi(name):
    # Use a breakpoint in the code line below to debug your script.
    print(f'Hi, {name}')  # Press Ctrl+F8 to toggle the breakpoint.


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    trial_mats_ml = np.load("..\\..\\..\\temp-data\\p_08-ml\\p_08-ml-kdf-mav-dmd.npz")
    trial_mats_hl = np.load("..\\..\\..\\temp-data\\p_08-hl\\p_08-hl-kdf-mav-dmd.npz")

    ml_arrays = list(trial_mats_ml.values())
    hl_arrays = list(trial_mats_hl.values())

    dmd_mat_list = ml_arrays + hl_arrays
    dmd_mat_labels = ["ML"]*len(ml_arrays)+["HL"]*len(hl_arrays)

    dmd_tot = len(dmd_mat_list)

    sims_dmd = np.zeros((dmd_tot, dmd_tot))
    sims_type = np.zeros((dmd_tot, dmd_tot))
    comparison_dmd = SimilarityTransformDist(iters=2000, lr=1e-3, score_method="wasserstein", wasserstein_compare='sv')

    for i, mi in enumerate(dmd_mat_list):
        print(i)
        for j, mj in enumerate(dmd_mat_list):
            stype = int(dmd_mat_labels[i] == dmd_mat_labels[j])
            sims_type[i, j] = sims_type[j, i] = stype
            if i == j:
                sims_type[i, i] = 2
            if j < i:
                continue
            sdmd = comparison_dmd.fit_score(mi, mj, zero_pad=True)
            print(i, j, sdmd)

            sims_dmd[i, j] = sims_dmd[j, i] = sdmd

    print("Finished Similarity Calculation")

    # sns.heatmap(sims_dmd[:,:], cmap="icefire")
    sns.heatmap(sims_dmd[:,:], cmap="icefire", vmax=10)
    plt.show()

    idx_exclude = []  # [26, 29, 34, 35]
    mask = np.ones(len(sims_dmd), bool)
    mask[idx_exclude] = False
    sns.heatmap(np.delete(np.delete(sims_dmd, ~mask, 0), ~mask,1), cmap="icefire")
    plt.show()

    df = pd.DataFrame()
    df['Model Type'] = np.array(dmd_mat_labels)[mask]
    reduced = MDS(dissimilarity='precomputed').fit_transform(np.delete(np.delete(sims_dmd, ~mask, 0), ~mask,1))
    df["0"] = reduced[:, 0]
    df["1"] = reduced[:, 1]

    palette = 'BuPu'
    sns.scatterplot(data=df, x="0", y="1", hue="Model Type", palette=palette)
    plt.xlabel(f"MDS 1")
    plt.ylabel(f"MDS 2")
    plt.tight_layout()
    plt.show()
# See PyCharm help at https://www.jetbrains.com/help/pycharm/
