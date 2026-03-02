import numpy as np
import xarray as xr
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset, Dataset
from sklearn.model_selection import StratifiedKFold
import matplotlib.pyplot as plt

# ── Configuration ─────────────────────────────────────────────────────────────
CONFIG = dict(
    # Data
    nc_path="stim_segmented_traces.nc",
    k_folds=8,

    # Training data mode
    augment=False,          # False = session-averaged (clean, fast)
                            # True  = mix-and-match augmentation
    n_augmented_samples=960,# only used when augment=True

    # Architecture
    rnn_type="GRU",         # "RNN", "LSTM", or "GRU"
    hidden_size=64,
    num_layers=1,
    dropout=0.5,

    # Training
    epochs=50,
    learning_rate=5e-4,
    batch_size=16,
)
# ─────────────────────────────────────────────────────────────────────────────

ANGLE_LABELS = [str(a) + "°" for a in range(0, 360, 30)]


# ── Data preparation ──────────────────────────────────────────────────────────
def load_raw(cfg):
    """Load all individual (non-averaged) trials. Used by both modes."""
    ds = xr.load_dataset(cfg["nc_path"])
    valid_mask  = ~np.isnan(ds["ids"].values)
    ds          = ds.isel(trial=valid_mask)
    raw_traces  = ds["trace"].values.astype(np.float32)  # (n_trials, stim_index, time)
    stim_angles = ds["stim"].values                       # (stim_index,)
    spine_ids   = ds["ids"].values                        # (n_trials,)
    unique_ids  = np.unique(spine_ids)

    spine_to_trials = {sid: np.where(spine_ids == sid)[0] for sid in unique_ids}
    n_spines = len(unique_ids)
    time     = raw_traces.shape[2]

    print(f"Unique spines : {n_spines}")
    print(f"Stim positions: {len(stim_angles)}  (12 angles × 8 repeats)")
    print(f"Time points   : {time}")
    print(f"Raw trials    : {len(spine_ids)}")
    print(f"Mode          : {'augmented mix-and-match' if cfg['augment'] else 'session-averaged'}\n")

    return raw_traces, stim_angles, unique_ids, spine_to_trials, n_spines, time


def make_averaged_xy(raw_traces, stim_angles, unique_ids, spine_to_trials):
    """Average sessions per spine → X: (stim_index, time, n_spines), y: (stim_index,)"""
    n_stim   = len(stim_angles)
    n_spines = len(unique_ids)
    time     = raw_traces.shape[2]
    X = np.empty((n_stim, time, n_spines), dtype=np.float32)
    for si, sid in enumerate(unique_ids):
        avg = raw_traces[spine_to_trials[sid]].mean(axis=0)  # (stim_index, time)
        X[:, :, si] = avg
    y = (stim_angles / 30).astype(np.int64)
    return X, y


# ── Augmented dataset (mix-and-match) ─────────────────────────────────────────
class MixMatchDataset(Dataset):
    def __init__(self, raw_traces, stim_angles, unique_ids, spine_to_trials,
                 train_stim_indices, n_samples, n_classes=12):
        self.raw_traces      = raw_traces
        self.unique_ids      = unique_ids
        self.spine_to_trials = spine_to_trials
        self.n_spines        = len(unique_ids)
        self.time            = raw_traces.shape[2]
        self.n_samples       = n_samples
        self.n_classes       = n_classes
        self.class_to_stim   = {}
        for cls in range(n_classes):
            angle     = cls * 30
            for_angle = np.where(stim_angles == angle)[0]
            self.class_to_stim[cls] = np.intersect1d(for_angle, train_stim_indices)

    def __len__(self):
        return self.n_samples

    def __getitem__(self, _):
        cls        = np.random.randint(self.n_classes)
        population = np.empty((self.time, self.n_spines), dtype=np.float32)
        for si, sid in enumerate(self.unique_ids):
            trial    = np.random.choice(self.spine_to_trials[sid])
            stim_pos = np.random.choice(self.class_to_stim[cls])
            population[:, si] = self.raw_traces[trial, stim_pos, :]
        return torch.tensor(population), torch.tensor(cls, dtype=torch.long)


# ── Loader construction ───────────────────────────────────────────────────────
def make_loaders(cfg, raw_traces, stim_angles, unique_ids, spine_to_trials,
                 X_avg, y_avg, train_idx, test_idx):
    test_ds     = TensorDataset(torch.tensor(X_avg[test_idx]), torch.tensor(y_avg[test_idx]))
    test_loader = DataLoader(test_ds, batch_size=cfg["batch_size"], shuffle=False)

    if cfg["augment"]:
        train_ds = MixMatchDataset(
            raw_traces, stim_angles, unique_ids, spine_to_trials,
            train_stim_indices=train_idx,
            n_samples=cfg["n_augmented_samples"],
        )
    else:
        train_ds = TensorDataset(torch.tensor(X_avg[train_idx]), torch.tensor(y_avg[train_idx]))

    train_loader = DataLoader(train_ds, batch_size=cfg["batch_size"], shuffle=True)
    return train_loader, test_loader


# ── Model ─────────────────────────────────────────────────────────────────────
class AngleRNN(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, dropout, rnn_type, n_classes=12):
        super().__init__()
        rnn_cls = {"RNN": nn.RNN, "LSTM": nn.LSTM, "GRU": nn.GRU}[rnn_type]
        self.rnn = rnn_cls(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=dropout if num_layers > 1 else 0.0,
            batch_first=True,
        )
        self.classifier = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(hidden_size, n_classes),
        )

    def forward(self, x):
        out, _ = self.rnn(x)
        return self.classifier(out[:, -1, :])


# ── Training & evaluation ─────────────────────────────────────────────────────
def train_epoch(model, loader, optimizer, criterion, device):
    model.train()
    total_loss, correct, n = 0, 0, 0
    for X_batch, y_batch in loader:
        X_batch, y_batch = X_batch.to(device), y_batch.to(device)
        optimizer.zero_grad()
        logits = model(X_batch)
        loss   = criterion(logits, y_batch)
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * len(y_batch)
        correct    += (logits.argmax(1) == y_batch).sum().item()
        n          += len(y_batch)
    return total_loss / n, correct / n


@torch.no_grad()
def eval_epoch(model, loader, criterion, device):
    model.eval()
    total_loss, correct, n = 0, 0, 0
    for X_batch, y_batch in loader:
        X_batch, y_batch = X_batch.to(device), y_batch.to(device)
        logits = model(X_batch)
        loss   = criterion(logits, y_batch)
        total_loss += loss.item() * len(y_batch)
        correct    += (logits.argmax(1) == y_batch).sum().item()
        n          += len(y_batch)
    return total_loss / n, correct / n


@torch.no_grad()
def collect_preds(model, loader, device):
    model.eval()
    all_preds, all_targets = [], []
    for X_batch, y_batch in loader:
        logits = model(X_batch.to(device))
        all_preds.append(logits.argmax(1).cpu().numpy())
        all_targets.append(y_batch.numpy())
    return np.concatenate(all_preds), np.concatenate(all_targets)


# ── Confusion matrix ──────────────────────────────────────────────────────────
def plot_confusion_matrix(all_preds, all_targets, fold_accs, augment,
                          save_path="confusion_matrix.png"):
    n_classes = 12
    cm = np.zeros((n_classes, n_classes), dtype=int)
    for t, p in zip(all_targets, all_preds):
        cm[t, p] += 1
    cm_norm = cm.astype(float) / cm.sum(axis=1, keepdims=True)

    mode_label = "augmented" if augment else "averaged"
    fig, axes  = plt.subplots(1, 2, figsize=(16, 6), facecolor="#0f0f0f")
    fig.suptitle(
        f"RNN Angle Classification ({mode_label}) — K-Fold Confusion\n"
        f"Mean acc: {np.mean(fold_accs):.3f} ± {np.std(fold_accs):.3f}  |  Chance: {1/12:.3f}",
        color="white", fontsize=13, y=1.01
    )

    for ax, data, title, fmt in [
        (axes[0], cm_norm, "Normalized (row %)", ".2f"),
        (axes[1], cm,      "Raw counts",         "d"),
    ]:
        ax.set_facecolor("#0f0f0f")
        im = ax.imshow(data, cmap="magma", aspect="auto",
                       vmin=0, vmax=(1.0 if fmt == ".2f" else None))
        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04).ax.yaxis.set_tick_params(color="white")

        for i in range(n_classes):
            for j in range(n_classes):
                val        = data[i, j]
                text       = f"{val:.2f}" if fmt == ".2f" else f"{val:d}"
                brightness = val / (data.max() + 1e-8)
                color      = "white" if brightness < 0.6 else "black"
                ax.text(j, i, text, ha="center", va="center", fontsize=7, color=color)

        ax.set_xticks(range(n_classes))
        ax.set_yticks(range(n_classes))
        ax.set_xticklabels(ANGLE_LABELS, rotation=45, ha="right", color="white", fontsize=8)
        ax.set_yticklabels(ANGLE_LABELS, color="white", fontsize=8)
        ax.set_xlabel("Predicted angle", color="white", fontsize=10)
        ax.set_ylabel("True angle",      color="white", fontsize=10)
        ax.set_title(title, color="#aaaaaa", fontsize=10, pad=8)
        ax.tick_params(colors="white")
        for spine in ax.spines.values():
            spine.set_edgecolor("#444444")

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close()
    print(f"Confusion matrix saved to {save_path}")


# ── Main ──────────────────────────────────────────────────────────────────────
def main(cfg=CONFIG):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}\n")

    raw_traces, stim_angles, unique_ids, spine_to_trials, n_spines, time = load_raw(cfg)
    X_avg, y_avg = make_averaged_xy(raw_traces, stim_angles, unique_ids, spine_to_trials)

    skf            = StratifiedKFold(n_splits=cfg["k_folds"], shuffle=True, random_state=42)
    criterion      = nn.CrossEntropyLoss()
    fold_best_accs = []
    all_preds, all_targets = [], []

    for fold, (train_idx, test_idx) in enumerate(skf.split(X_avg, y_avg), 1):
        print(f"{'─'*60}")
        print(f"Fold {fold}/{cfg['k_folds']}  |  train: {len(train_idx)}  test: {len(test_idx)}")
        print(f"{'─'*60}")

        train_loader, test_loader = make_loaders(
            cfg, raw_traces, stim_angles, unique_ids, spine_to_trials,
            X_avg, y_avg, train_idx, test_idx
        )

        model = AngleRNN(
            input_size=n_spines,
            hidden_size=cfg["hidden_size"],
            num_layers=cfg["num_layers"],
            dropout=cfg["dropout"],
            rnn_type=cfg["rnn_type"],
        ).to(device)

        optimizer = torch.optim.Adam(model.parameters(), lr=cfg["learning_rate"])
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode="max", patience=5, factor=0.5
        )

        # Save state from epoch 1 unconditionally to avoid None on load
        best_acc   = -1.0
        best_state = None

        for epoch in range(1, cfg["epochs"] + 1):
            train_loss, train_acc = train_epoch(model, train_loader, optimizer, criterion, device)
            test_loss,  test_acc  = eval_epoch(model, test_loader,  criterion, device)
            scheduler.step(test_acc)

            if test_acc > best_acc:
                best_acc   = test_acc
                best_state = {k: v.clone() for k, v in model.state_dict().items()}
                torch.save(best_state, f"best_rnn_fold{fold}.pt")

            print(
                f"  Epoch {epoch:03d}/{cfg['epochs']} | "
                f"Train loss: {train_loss:.4f}  acc: {train_acc:.3f} | "
                f"Test  loss: {test_loss:.4f}  acc: {test_acc:.3f}"
                + (" ✓" if test_acc == best_acc else "")
            )

        model.load_state_dict(best_state)
        preds, targets = collect_preds(model, test_loader, device)
        all_preds.append(preds)
        all_targets.append(targets)
        fold_best_accs.append(best_acc)
        print(f"  → Best acc this fold: {best_acc:.3f}\n")

    print(f"{'═'*60}")
    print(f"K-Fold Results ({cfg['k_folds']} folds) — {'augmented' if cfg['augment'] else 'averaged'}")
    print(f"{'═'*60}")
    for i, acc in enumerate(fold_best_accs, 1):
        print(f"  Fold {i}: {acc:.3f}")
    print(f"  Mean ± Std: {np.mean(fold_best_accs):.3f} ± {np.std(fold_best_accs):.3f}")
    print(f"  Chance level: {1/12:.3f}")

    plot_confusion_matrix(
        np.concatenate(all_preds),
        np.concatenate(all_targets),
        fold_best_accs,
        augment=cfg["augment"],
        save_path="confusion_matrix.png"
    )


if __name__ == "__main__":
    main()