"""
dpca_visualize.py
─────────────────
Runs dPCA and produces 5 figures:
  1. 4-panel 2D comparison (means / smoothed / scatter / means+clouds)
  2. Condition means 3D
  3. Smoothed trajectories 3D
  4. Mean dot scatter 3D
  5. Condition means + trial clouds 3D
"""

import warnings
warnings.filterwarnings("ignore")

import numpy as np
import xarray as xr
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from scipy.ndimage import gaussian_filter1d
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401

# ─────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────

DATA_PATH      = "../Data/b6_spine_traces.nc"
N_LATENT       = 10
N_PLOT         = 3
SMOOTH_SIGMA   = 5     # frames, for Gaussian smoothing

TRAJ_ALPHA     = 0.20
TRAJ_LW        = 0.8
MEAN_TRAJ_LW   = 2.2
MEAN_DOT_SIZE  = 60
CLOUD_ALPHA    = 0.25

# ─────────────────────────────────────────────
# DATA & dPCA
# ─────────────────────────────────────────────

def load_data():
    ds         = xr.open_dataset(DATA_PATH)
    spine_rois = ds.roi.where(ds.structure == "spine", drop=True)
    data       = ds["angle"].sel(roi=spine_rois).values
    angles     = ds["stimulus_angle"].values
    print(f"Data: trials={data.shape[0]}, frames={data.shape[1]}, neurons={data.shape[2]}")
    return data, angles


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
    latent   = np.transpose(Z["s"], (1, 2, 0))   # (T, F, n_latent)
    return latent, unique_angles

# ─────────────────────────────────────────────
# DERIVED REPRESENTATIONS
# ─────────────────────────────────────────────

def condition_means(latent, angles, unique_angles):
    """Mean trajectory per angle: (n_angles, F, n_latent)"""
    return np.stack([latent[angles == a].mean(axis=0) for a in unique_angles])


def smoothed(latent):
    """Gaussian-smooth each trial along time axis."""
    return gaussian_filter1d(latent, sigma=SMOOTH_SIGMA, axis=1)

# ─────────────────────────────────────────────
# COLOR HELPERS
# ─────────────────────────────────────────────

def make_colors(unique_angles):
    cmap      = cm.get_cmap("hsv")
    n         = len(unique_angles)
    color_map = {a: cmap(i / n) for i, a in enumerate(unique_angles)}
    return color_map


def trial_colors(angles, color_map):
    return [color_map[a] for a in angles]


def legend_handles(color_map, unique_angles):
    return [
        plt.Line2D([0], [0], marker="o", color="w",
                   markerfacecolor=color_map[a], markersize=7,
                   label=f"{int(a)}°")
        for a in unique_angles
    ]

# ─────────────────────────────────────────────
# 2D PANEL HELPERS
# ─────────────────────────────────────────────

def _draw_trajectory(ax, traj, color, alpha, lw, arrow=True):
    """Draw a single 2D trajectory with optional midpoint arrow."""
    x, y = traj[:, 0], traj[:, 1]
    ax.plot(x, y, color=color, alpha=alpha, linewidth=lw)
    if arrow and len(x) > 2:
        mid = len(x) // 2
        ax.annotate("", xy=(x[mid+1], y[mid+1]), xytext=(x[mid], y[mid]),
                    arrowprops=dict(arrowstyle="->", color=color,
                                    lw=0.8, alpha=min(alpha * 2, 1.0)))


def _draw_mean_dot(ax, traj, color):
    ax.scatter(traj[:, 0].mean(), traj[:, 1].mean(),
               color=color, s=MEAN_DOT_SIZE, alpha=0.95,
               edgecolors="white", linewidths=0.5, zorder=5)


def _finish_ax(ax, title, xlabel="dPC 1", ylabel="dPC 2"):
    ax.set_xlabel(xlabel, fontsize=9)
    ax.set_ylabel(ylabel, fontsize=9)
    ax.set_title(title, fontsize=10)
    ax.grid(alpha=0.2)
    ax.set_aspect("equal", adjustable="datalim")

# ─────────────────────────────────────────────
# 3D HELPERS
# ─────────────────────────────────────────────

def _draw_3d_traj(ax, traj, color, alpha, lw):
    ax.plot(traj[:, 0], traj[:, 1], traj[:, 2],
            color=color, alpha=alpha, linewidth=lw)


def _draw_3d_dot(ax, traj, color, size=MEAN_DOT_SIZE):
    ax.scatter(traj[:, 0].mean(), traj[:, 1].mean(), traj[:, 2].mean(),
               color=color, s=size, alpha=0.95,
               edgecolors="white", linewidths=0.4, zorder=5)


def _finish_ax3d(ax, title):
    ax.set_xlabel("dPC 1", labelpad=5)
    ax.set_ylabel("dPC 2", labelpad=5)
    ax.set_zlabel("dPC 3", labelpad=5)
    ax.set_title(title, fontsize=11)

# ─────────────────────────────────────────────
# FIGURE 1 — 2D FOUR-PANEL COMPARISON
# ─────────────────────────────────────────────

def plot_comparison_2d(latent, angles, unique_angles, color_map):
    cmeans  = condition_means(latent, angles, unique_angles)
    smooth  = smoothed(latent)
    tcolors = trial_colors(angles, color_map)
    acolors = [color_map[a] for a in unique_angles]

    fig, axes = plt.subplots(1, 4, figsize=(20, 5))
    fig.suptitle("dPCA stimulus subspace — dPC1 vs dPC2  (all styles)", fontsize=13)

    # ── Panel A: condition means ──────────────────────────────────
    ax = axes[0]
    for i, a in enumerate(unique_angles):
        traj = cmeans[i, :, :2]
        _draw_trajectory(ax, traj, acolors[i], alpha=0.9, lw=MEAN_TRAJ_LW)
        _draw_mean_dot(ax, traj, acolors[i])
    _finish_ax(ax, "A: Condition means")

    # ── Panel B: smoothed individual trials ───────────────────────
    ax = axes[1]
    for t in range(latent.shape[0]):
        traj = smooth[t, :, :2]
        _draw_trajectory(ax, traj, tcolors[t], alpha=TRAJ_ALPHA * 1.5, lw=TRAJ_LW)
        _draw_mean_dot(ax, traj, tcolors[t])
    _finish_ax(ax, f"B: Smoothed trials (σ={SMOOTH_SIGMA} frames)")

    # ── Panel C: mean dot scatter only ───────────────────────────
    ax = axes[2]
    for t in range(latent.shape[0]):
        traj = latent[t, :, :2]
        _draw_mean_dot(ax, traj, tcolors[t])
    _finish_ax(ax, "C: Trial mean scatter")

    # ── Panel D: condition means + trial clouds ───────────────────
    ax = axes[3]
    for t in range(latent.shape[0]):
        traj = latent[t, :, :2]
        ax.scatter(traj[:, 0].mean(), traj[:, 1].mean(),
                   color=tcolors[t], s=25, alpha=CLOUD_ALPHA,
                   edgecolors="none", zorder=2)
    for i, a in enumerate(unique_angles):
        traj = cmeans[i, :, :2]
        _draw_trajectory(ax, traj, acolors[i], alpha=0.95, lw=MEAN_TRAJ_LW)
        _draw_mean_dot(ax, traj, acolors[i])
    _finish_ax(ax, "D: Means + trial clouds")

    # Shared legend on last panel
    axes[3].legend(handles=legend_handles(color_map, unique_angles),
                   title="Angle", fontsize=7, title_fontsize=8,
                   loc="best", ncol=2, framealpha=0.7)

    plt.tight_layout()
    plt.savefig("dpca_comparison_2d.png", dpi=150)
    plt.show()
    print("Saved → dpca_comparison_2d.png")

# ─────────────────────────────────────────────
# FIGURES 2–5 — 3D VERSIONS
# ─────────────────────────────────────────────

def plot_3d_all(latent, angles, unique_angles, color_map):
    cmeans  = condition_means(latent, angles, unique_angles)
    smooth  = smoothed(latent)
    tcolors = trial_colors(angles, color_map)
    acolors = [color_map[a] for a in unique_angles]

    configs = [
        ("A: Condition means",          "dpca_3d_means.png"),
        ("B: Smoothed trials",          "dpca_3d_smooth.png"),
        ("C: Trial mean scatter",       "dpca_3d_scatter.png"),
        ("D: Means + trial clouds",     "dpca_3d_clouds.png"),
    ]

    for idx, (title, fname) in enumerate(configs):
        fig = plt.figure(figsize=(8, 6))
        ax  = fig.add_subplot(111, projection="3d")

        if idx == 0:   # condition means
            for i, a in enumerate(unique_angles):
                _draw_3d_traj(ax, cmeans[i], acolors[i], alpha=0.9, lw=MEAN_TRAJ_LW)
                _draw_3d_dot(ax, cmeans[i], acolors[i])

        elif idx == 1:  # smoothed trials
            for t in range(latent.shape[0]):
                _draw_3d_traj(ax, smooth[t], tcolors[t], alpha=TRAJ_ALPHA * 1.5, lw=TRAJ_LW)
                _draw_3d_dot(ax, smooth[t], tcolors[t])

        elif idx == 2:  # scatter only
            for t in range(latent.shape[0]):
                _draw_3d_dot(ax, latent[t], tcolors[t])

        elif idx == 3:  # means + clouds
            for t in range(latent.shape[0]):
                m = latent[t].mean(axis=0)
                ax.scatter(m[0], m[1], m[2], color=tcolors[t],
                           s=20, alpha=CLOUD_ALPHA, edgecolors="none")
            for i, a in enumerate(unique_angles):
                _draw_3d_traj(ax, cmeans[i], acolors[i], alpha=0.95, lw=MEAN_TRAJ_LW)
                _draw_3d_dot(ax, cmeans[i], acolors[i])

        _finish_ax3d(ax, title)
        ax.legend(handles=legend_handles(color_map, unique_angles),
                  title="Angle", fontsize=7, title_fontsize=8,
                  loc="upper left", ncol=2, framealpha=0.7)

        plt.tight_layout()
        plt.savefig(fname, dpi=150)
        plt.show()
        print(f"Saved → {fname}")

# ─────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────

def main():
    data, angles          = load_data()
    latent, unique_angles = run_dpca(data, angles, N_LATENT)
    latent                = latent[:, :, :N_PLOT]
    color_map             = make_colors(unique_angles)

    plot_comparison_2d(latent, angles, unique_angles, color_map)
    plot_3d_all(latent, angles, unique_angles, color_map)


if __name__ == "__main__":
    main()