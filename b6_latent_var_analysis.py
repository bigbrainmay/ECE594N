import numpy as np
import xarray as xr
import matplotlib.pyplot as plt

from sklearn.decomposition import PCA, FactorAnalysis
from sklearn.model_selection import cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC

# optional models
from dPCA.dPCA import dPCA
from elephant.gpfa import GPFA


# ─────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────

DATA_PATH = "../Data/b6_spine_traces.nc"

LATENT_MODEL = "gpfa"   # pca | fa | gpfa | dpca
CLASSIFIER = "svm"

N_LATENT = 6


# ─────────────────────────────────────────────
# LOAD DATA
# ─────────────────────────────────────────────

def load_data():

    ds = xr.open_dataset(DATA_PATH)

    spine_rois = ds.roi.where(ds.structure == "spine", drop=True)

    data = ds["angle"].sel(roi=spine_rois).values
    angles = ds["stimulus_angle"].values

    return data, angles


# ─────────────────────────────────────────────
# PCA / FA
# ─────────────────────────────────────────────

def run_pca_fa(data, model):

    T, F, N = data.shape

    X = data.reshape(T * F, N)

    latent = model.fit_transform(X)

    latent = latent.reshape(T, F, -1)

    return latent


# ─────────────────────────────────────────────
# GPFA
# ─────────────────────────────────────────────
def run_gpfa(data, n_latent=N_LATENT):
    from elephant.gpfa import GPFA
    import quantities as pq
    import neo

    # Convert each trial to a neo.AnalogSignal (frames × neurons)
    gpfa_data = [
        neo.AnalogSignal(
            data[i],              # (frames, neurons)
            units='dimensionless',
            sampling_rate=1 * pq.Hz
        )
        for i in range(data.shape[0])
    ]

    model = GPFA(bin_size=1 * pq.s, x_dim=n_latent)
    latent_list = model.fit_transform(gpfa_data)  # list of (n_latent, frames)

    latent = np.stack([latent_list[i].T for i in range(len(latent_list))])  # (T, F, n_latent)
    return latent

# ─────────────────────────────────────────────
# dPCA
# ─────────────────────────────────────────────
def run_dpca(data, angles):

    unique_angles = np.sort(np.unique(angles))

    T, F, N = data.shape
    C = len(unique_angles)

    # Compute condition averages (needed for fitting)
    X = np.zeros((N, C, F))

    for i, a in enumerate(unique_angles):
        trials = data[angles == a]
        X[:, i, :] = trials.mean(axis=0).T

    # Fit dPCA
    dpca = dPCA(labels='st', n_components=N_LATENT)
    dpca.protect = ['t']  # protect time

    dpca.fit(X)

    # Now project ALL trials
    X_trials = np.transpose(data, (2, 0, 1))  # neurons × trials × time

    Z = dpca.transform(X_trials)

    # Z is a dict of components ('s','t','st')
    Zs = Z['s']  # stimulus-related components

    # reshape back to (trial, frame, latent)
    latent = np.transpose(Zs, (1, 2, 0))

    return latent

# ─────────────────────────────────────────────
# FEATURE EXTRACTION
# ─────────────────────────────────────────────

def trial_features(latent):

    return latent.mean(axis=1)


# ─────────────────────────────────────────────
# CLASSIFIER
# ─────────────────────────────────────────────

def build_classifier():

    if CLASSIFIER == "logreg":
        clf = LogisticRegression(max_iter=1000)

    else:
        clf = SVC(kernel="rbf")

    return Pipeline([
        ("scale", StandardScaler()),
        ("clf", clf)
    ])


def decode(features, labels):

    clf = build_classifier()

    scores = cross_val_score(clf, features, labels, cv=5)

    print("Decoding accuracy:", scores.mean())
    print("CV scores:", scores)


# ─────────────────────────────────────────────
# PLOT TRAJECTORIES
# ─────────────────────────────────────────────

def plot_latent(latent, angles):

    T = latent.shape[0]

    plt.figure()

    for trial in range(T):

        plt.plot(
            latent[trial,:,0],
            latent[trial,:,1],
            alpha=0.4
        )

    plt.xlabel("latent 1")
    plt.ylabel("latent 2")

    plt.title("latent trajectories")

    plt.show()


# ─────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────

def main():

    data, angles = load_data()

    print("data shape:", data.shape)

    if LATENT_MODEL == "pca":

        model = PCA(n_components=N_LATENT)
        latent = run_pca_fa(data, model)

    elif LATENT_MODEL == "fa":

        model = FactorAnalysis(n_components=N_LATENT)
        latent = run_pca_fa(data, model)

    elif LATENT_MODEL == "gpfa":

        latent = run_gpfa(data)

    elif LATENT_MODEL == "dpca":

        latent = run_dpca(data, angles)

        # dPCA returns condition averages
        features = trial_features(latent)
        decode(features, angles)
        plot_latent(latent, angles)

        return

    else:
        raise ValueError("Unknown model")

    plot_latent(latent, angles)

    features = trial_features(latent)

    decode(features, angles)


if __name__ == "__main__":
    main()