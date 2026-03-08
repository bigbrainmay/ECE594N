"""
dpca_visualize.py
─────────────────
Runs dPCA with N_LATENT components, takes the top 3 stimulus-related
components, and plots all trials as trajectories colored by stimulus angle.

Produces two figures:
  1. dpca_trajectories_3d.png  — interactive 3D plot
  2. dpca_trajectories_2d.png  — three 2D projection panels (1v2, 1v3, 2v3)
"""

import warnings
warnings.filterwarnings("ignore")

import numpy as np
import xarray as xr
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401

# ─────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────

DATA_PATH = "../Data/b6_spine_traces.nc"
N_LATENT  = 10   # total dPCA components to fit
N_PLOT    = 3    # top stimulus components to visualise

TRAJ_ALPHA  = 0.25   # faint trajectory lines
TRAJ_LW     = 0.8
MEAN_SIZE   = 60     # bold mean dot size
MEAN_ALPHA  = 0.95

# ─────────────────────────────────────────────
# DATA
# ─────────────────────────────────────────────

def load_data():
    ds         = xr.open_dataset(DATA_PATH)
    spine_rois = ds.roi.where(ds.structure == "spine", drop=True)
    data       = ds["angle"].sel(roi=spine_rois).values   # (T, F, N)
    angles     = ds["stimulus_angle"].values
    print(f"Data: trials={data.shape[0]}, frames={data.shape[1]}, neurons={data.shape[2]}")
    return data, angles

# ─────────────────────────────────────────────
# dPCA
# ─────────────────────────────────────────────

def run_dpca(data, angles, n_latent):
    from dPCA.dPCA import dPCA as dPCAModel

    T, F, N       = data.shape
    unique_angles = np.sort(np.unique(angles))
    C             = len(unique_angles)

    # Condition averages: (neurons, conditions, frames)
    X = np.zeros((N, C, F))
    for i, a in enumerate(unique_angles):
        trials      = data[angles == a]
        X[:, i, :] = trials.mean(axis=0).T

    dpca         = dPCAModel(labels="st", n_components=n_latent)
    dpca.protect = ["t"]
    dpca.fit(X)

    # Project all trials → stimulus components only
    X_trials = np.transpose(data, (2, 0, 1))   # (N, T, F)
    Z        = dpca.transform(X_trials)
    Zs       = Z["s"]                           # (n_latent, T, F)
    latent   = np.transpose(Zs, (1, 2, 0))      # (T, F, n_latent)
    return latent, unique_angles

# ─────────────────────────────────────────────
# COLOR HELPERS
# ─────────────────────────────────────────────

def angle_colors(angles, unique_angles):
    """Map each trial's angle to a color using a circular colormap."""
    cmap      = cm.get_cmap("hsv")
    n         = len(unique_angles)
    color_map = {a: cmap(i / n) for i, a in enumerate(unique_angles)}
    return np.array([color_map[a] for a in angles]), color_map


def angle_legend(ax, color_map, unique_angles):
    """Add a compact angle legend."""
    handles = [
        plt.Line2D([0], [0], marker="o", color="w",
                   markerfacecolor=color_map[a], markersize=7,
                   label=f"{int(a)}°")
        for a in unique_angles
    ]
    ax.legend(handles=handles, title="Angle", fontsize=7,
              title_fontsize=8, loc="best", ncol=2,
              framealpha=0.7, handletextpad=0.3)

# ─────────────────────────────────────────────
# 3D PLOT
# ─────────────────────────────────────────────

def plot_3d(latent, angles, unique_angles, colors, color_map):
    fig = plt.figure(figsize=(9, 7))
    ax  = fig.add_subplot(111, projection="3d")

    c0, c1, c2 = 0, 1, 2

    for trial in range(latent.shape[0]):
        x, y, z = latent[trial, :, c0], latent[trial, :, c1], latent[trial, :, c2]
        col = colors[trial]

        # Faint trajectory
        ax.plot(x, y, z, color=col, alpha=TRAJ_ALPHA, linewidth=TRAJ_LW)

        # Bold mean dot
        ax.scatter(x.mean(), y.mean(), z.mean(),
                   color=col, s=MEAN_SIZE, alpha=MEAN_ALPHA,
                   edgecolors="white", linewidths=0.4, zorder=5)

    ax.set_xlabel(f"dPC 1", labelpad=6)
    ax.set_ylabel(f"dPC 2", labelpad=6)
    ax.set_zlabel(f"dPC 3", labelpad=6)
    ax.set_title(f"dPCA stimulus trajectories (top 3 of {N_LATENT} components)\n"
                 f"n={latent.shape[0]} trials, {len(unique_angles)} angles", fontsize=11)

    angle_legend(ax, color_map, unique_angles)

    plt.tight_layout()
    plt.savefig("dpca_trajectories_3d.png", dpi=150)
    plt.show()
    print("Saved → dpca_trajectories_3d.png")

# ─────────────────────────────────────────────
# 2D PROJECTION PANELS
# ─────────────────────────────────────────────

def plot_2d(latent, angles, unique_angles, colors, color_map):
    pairs  = [(0, 1), (0, 2), (1, 2)]
    labels = ["dPC 1", "dPC 2", "dPC 3"]

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    fig.suptitle(f"dPCA stimulus trajectories — 2D projections\n"
                 f"n={latent.shape[0]} trials, {len(unique_angles)} angles",
                 fontsize=12)

    for ax, (ci, cj) in zip(axes, pairs):
        for trial in range(latent.shape[0]):
            x   = latent[trial, :, ci]
            y   = latent[trial, :, cj]
            col = colors[trial]

            # Faint trajectory
            ax.plot(x, y, color=col, alpha=TRAJ_ALPHA, linewidth=TRAJ_LW)

            # Arrow at midpoint to show direction of time
            mid = len(x) // 2
            ax.annotate("", xy=(x[mid+1], y[mid+1]), xytext=(x[mid], y[mid]),
                        arrowprops=dict(arrowstyle="->", color=col,
                                        lw=0.6, alpha=TRAJ_ALPHA * 1.5))

            # Bold mean dot
            ax.scatter(x.mean(), y.mean(),
                       color=col, s=MEAN_SIZE, alpha=MEAN_ALPHA,
                       edgecolors="white", linewidths=0.4, zorder=5)

        ax.set_xlabel(labels[ci])
        ax.set_ylabel(labels[cj])
        ax.set_title(f"{labels[ci]} vs {labels[cj]}")
        ax.grid(alpha=0.2)
        ax.set_aspect("equal", adjustable="datalim")

    # Single shared legend on last panel
    angle_legend(axes[-1], color_map, unique_angles)

    plt.tight_layout()
    plt.savefig("dpca_trajectories_2d.png", dpi=150)
    plt.show()
    print("Saved → dpca_trajectories_2d.png")

# ─────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────

def main():
    data, angles          = load_data()
    latent, unique_angles = run_dpca(data, angles, N_LATENT)

    # Keep only top N_PLOT stimulus components
    latent = latent[:, :, :N_PLOT]

    colors, color_map = angle_colors(angles, unique_angles)

    plot_3d(latent, angles, unique_angles, colors, color_map)
    plot_2d(latent, angles, unique_angles, colors, color_map)


if __name__ == "__main__":
    main()