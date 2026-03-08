"""
benchmark_latent_models.py
──────────────────────────
Sweeps over latent models (PCA, FA, dPCA) and a range of N_LATENT values,
extracts windowed temporal features, runs 5-fold CV decoding, and
produces a summary table + heatmap/line figure, plus a dPCA fine sweep
with d-prime calculation.
"""

import warnings
warnings.filterwarnings("ignore")

import numpy as np
import xarray as xr
import matplotlib.pyplot as plt
import pandas as pd
from scipy.stats import norm

from sklearn.decomposition import PCA, FactorAnalysis
from sklearn.model_selection import cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

# ─────────────────────────────────────────────
# CONFIG  ← edit here
# ─────────────────────────────────────────────

DATA_PATH = "../Data/b6_spine_traces.nc"

LATENT_DIMS = {
    "pca":  [2, 4, 6, 8, 10, 12],
    "fa":   [2, 4, 6, 8, 10, 12],
    "dpca": [2, 4, 6, 8, 10, 12, 16, 20, 24, 28, 32],
}

MODELS     = ["pca", "fa", "dpca"]
CLASSIFIER = "svm"   # svm | logreg
CV_FOLDS   = 5

# Temporal windows as fractions of total frames.
# Set to None to use a single whole-trial mean.
TIME_WINDOWS = [(0.0, 0.33), (0.33, 0.66), (0.66, 1.0)]

OUTPUT_TABLE  = "benchmark_results.csv"
OUTPUT_FIGURE = "benchmark_heatmap.png"

# ─────────────────────────────────────────────
# DATA LOADING
# ─────────────────────────────────────────────

def load_data():
    ds         = xr.open_dataset(DATA_PATH)
    spine_rois = ds.roi.where(ds.structure == "spine", drop=True)
    data       = ds["angle"].sel(roi=spine_rois).values
    angles     = ds["stimulus_angle"].values
    print(f"Data loaded: trials={data.shape[0]}, frames={data.shape[1]}, neurons={data.shape[2]}")
    return data, angles

# ─────────────────────────────────────────────
# LATENT MODELS
# ─────────────────────────────────────────────

def run_pca_fa(data, model):
    T, F, N = data.shape
    X       = data.reshape(T * F, N)
    latent  = model.fit_transform(X)
    return latent.reshape(T, F, -1)


def run_dpca(data, angles, n_latent):
    from dPCA.dPCA import dPCA as dPCAModel

    T, F, N       = data.shape
    unique_angles = np.sort(np.unique(angles))
    C             = len(unique_angles)

    X = np.zeros((N, C, F))
    for i, a in enumerate(unique_angles):
        trials      = data[angles == a]
        X[:, i, :] = trials.mean(axis=0).T

    dpca         = dPCAModel(labels="st", n_components=n_latent)
    dpca.protect = ["t"]
    dpca.fit(X)

    X_trials = np.transpose(data, (2, 0, 1))
    Z        = dpca.transform(X_trials)
    Zs       = Z["s"]
    latent   = np.transpose(Zs, (1, 2, 0))
    return latent

# ─────────────────────────────────────────────
# FEATURE EXTRACTION
# ─────────────────────────────────────────────

def trial_features(latent):
    T, F, L = latent.shape
    if TIME_WINDOWS is None:
        return latent.mean(axis=1)
    segments = []
    for (t0, t1) in TIME_WINDOWS:
        f0 = int(np.floor(t0 * F))
        f1 = int(np.ceil(t1 * F))
        segments.append(latent[:, f0:f1, :].mean(axis=1))
    return np.concatenate(segments, axis=1)

# ─────────────────────────────────────────────
# CLASSIFIER
# ─────────────────────────────────────────────

def build_classifier():
    from sklearn.linear_model import LogisticRegression
    if CLASSIFIER == "logreg":
        clf = LogisticRegression(max_iter=1000)
    else:
        clf = SVC(kernel="rbf")
    return Pipeline([("scale", StandardScaler()), ("clf", clf)])


def decode(latent, labels):
    features = trial_features(latent)
    clf      = build_classifier()
    scores   = cross_val_score(clf, features, labels, cv=CV_FOLDS)
    return scores.mean(), scores.std()

# ─────────────────────────────────────────────
# BENCHMARK LOOP
# ─────────────────────────────────────────────

def run_benchmark(data, angles):
    results = []
    for model_name in MODELS:
        print(f"\n── Model: {model_name.upper()} ──")
        for n in LATENT_DIMS[model_name]:
            print(f"  n_latent={n} ... ", end="", flush=True)
            try:
                if model_name == "pca":
                    latent = run_pca_fa(data, PCA(n_components=n))
                elif model_name == "fa":
                    latent = run_pca_fa(data, FactorAnalysis(n_components=n))
                elif model_name == "dpca":
                    latent = run_dpca(data, angles, n)
                mean_acc, std_acc = decode(latent, angles)
                print(f"acc={mean_acc:.3f} ± {std_acc:.3f}")
            except Exception as e:
                mean_acc, std_acc = np.nan, np.nan
                print(f"FAILED ({e})")
            results.append({
                "model":    model_name.upper(),
                "n_latent": n,
                "mean_acc": mean_acc,
                "std_acc":  std_acc,
            })
    return pd.DataFrame(results)

# ─────────────────────────────────────────────
# REPORTING
# ─────────────────────────────────────────────

def save_table(df):
    pivot = df.pivot(index="model", columns="n_latent", values="mean_acc").round(3)
    print("\n\n════════════════════════════════════════════")
    print("  DECODING ACCURACY (mean over 5-fold CV)")
    print(f"  Chance level: {1/12:.1%}  (12 angles)")
    print("════════════════════════════════════════════")
    print(pivot.to_string())
    pivot.to_csv(OUTPUT_TABLE)
    print(f"\nTable saved → {OUTPUT_TABLE}")
    return pivot


def save_figure(df):
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    ax     = axes[0]
    pca_fa = df[df["model"].isin(["PCA", "FA"])]
    pivot_pf = pca_fa.pivot(index="model", columns="n_latent", values="mean_acc")
    mat    = pivot_pf.values.astype(float)
    vmin, vmax = np.nanmin(mat), np.nanmax(mat)
    im = ax.imshow(mat, aspect="auto", cmap="viridis", vmin=vmin, vmax=vmax)
    ax.set_xticks(range(len(pivot_pf.columns)))
    ax.set_xticklabels(pivot_pf.columns)
    ax.set_yticks(range(len(pivot_pf.index)))
    ax.set_yticklabels(pivot_pf.index)
    ax.set_xlabel("N latent dimensions")
    ax.set_title(f"Accuracy — PCA & FA ({CLASSIFIER.upper()}, {CV_FOLDS}-fold CV)")
    for r in range(mat.shape[0]):
        for c in range(mat.shape[1]):
            v = mat[r, c]
            if not np.isnan(v):
                lum = 0.2 if v > vmin + (vmax - vmin) * 0.6 else 0.95
                ax.text(c, r, f"{v:.2f}", ha="center", va="center", fontsize=8, color=str(lum))
    plt.colorbar(im, ax=ax, label="Accuracy")

    ax2    = axes[1]
    colors = {"PCA": "#4C9BE8", "FA": "#E8834C", "DPCA": "#56C17A"}
    for model_name in df["model"].unique():
        row = df[df["model"] == model_name].sort_values("n_latent")
        ax2.errorbar(row["n_latent"], row["mean_acc"], yerr=row["std_acc"],
                     marker="o", label=model_name, capsize=3, linewidth=2,
                     color=colors.get(model_name))
    ax2.axhline(1/12, color="gray", linestyle="--", linewidth=1, label="Chance (8.3%)")
    ax2.set_xlabel("N latent dimensions")
    ax2.set_ylabel("Decoding accuracy")
    ax2.set_title("Accuracy vs. N latent (all models)")
    ax2.legend(fontsize=9)
    ax2.set_ylim(0, 1.05)
    ax2.grid(alpha=0.3)

    plt.tight_layout()
    plt.savefig(OUTPUT_FIGURE, dpi=150)
    plt.show()
    print(f"Figure saved → {OUTPUT_FIGURE}")

# ─────────────────────────────────────────────
# dPCA FINE SWEEP + d'
# ─────────────────────────────────────────────

def dpca_fine_sweep(data, angles):
    dims  = list(range(10, 21))
    means = []
    stds  = []

    print("\n── dPCA fine sweep (10–20) ──")
    for n in dims:
        print(f"  n_latent={n} ... ", end="", flush=True)
        try:
            latent            = run_dpca(data, angles, n)
            mean_acc, std_acc = decode(latent, angles)
            print(f"acc={mean_acc:.3f} ± {std_acc:.3f}")
        except Exception as e:
            mean_acc, std_acc = np.nan, np.nan
            print(f"FAILED ({e})")
        means.append(mean_acc)
        stds.append(std_acc)

    means = np.array(means)
    stds  = np.array(stds)

    # d' at best n
    best_idx  = np.nanargmax(means)
    best_n    = dims[best_idx]
    best_acc  = means[best_idx]
    n_classes = len(np.unique(angles))
    chance    = 1.0 / n_classes
    dprime    = norm.ppf(np.clip(best_acc, 1e-6, 1-1e-6)) - norm.ppf(np.clip(chance, 1e-6, 1-1e-6))

    print(f"\nBest: n_latent={best_n}, acc={best_acc:.3f}, d'={dprime:.3f}")

    # Plot
    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.errorbar(dims, means, yerr=stds, marker="o", color="#56C17A",
                linewidth=2, capsize=4, label="dPCA accuracy")
    ax.axhline(chance, color="gray", linestyle="--", linewidth=1,
               label=f"Chance ({chance:.1%})")
    ax.annotate(
        f"n={best_n}\nacc={best_acc:.2f}\nd'={dprime:.2f}",
        xy=(best_n, best_acc),
        xytext=(best_n + 0.4, best_acc - 0.07),
        arrowprops=dict(arrowstyle="->", color="black"),
        fontsize=9,
        bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="gray")
    )
    ax.set_xlabel("N latent dimensions (dPCA)")
    ax.set_ylabel("Decoding accuracy")
    ax.set_title(f"dPCA fine sweep — {CV_FOLDS}-fold CV, {n_classes} angles")
    ax.set_xticks(dims)
    ax.set_ylim(0, 1.05)
    ax.legend(fontsize=9)
    ax.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig("dpca_fine_sweep.png", dpi=150)
    plt.show()
    print("Figure saved → dpca_fine_sweep.png")

# ─────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────

def main():
    data, angles = load_data()
    df           = run_benchmark(data, angles)
    pivot        = save_table(df)
    save_figure(df)
    dpca_fine_sweep(data, angles)


if __name__ == "__main__":
    main()