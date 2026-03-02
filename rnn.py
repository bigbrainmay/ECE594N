import numpy as np
import xarray as xr
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from sklearn.model_selection import StratifiedKFold

# ── Configuration ─────────────────────────────────────────────────────────────
CONFIG = dict(
    # Data
    nc_path="stim_segmented_traces.nc",
    k_folds=8,
    n_augmented_samples=96*5,    # synthetic training samples per fold (10× original 96)

    # Architecture
    rnn_type="RNN",             # "RNN", "LSTM", or "GRU"
    hidden_size=32,
    num_layers=1,
    dropout=0.5,

    # Training
    epochs=50,
    learning_rate=5e-4,
    batch_size=32,
)
# ─────────────────────────────────────────────────────────────────────────────


# ── Data preparation ──────────────────────────────────────────────────────────
def load_data(cfg):
    """
    Returns:
        raw_traces       : (n_trials, n_stim_index, time)  — all individual trials, NOT averaged
        stim_angles      : (n_stim_index,)                 — angle per stim_index position
        unique_ids       : sorted array of unique spine IDs
        spine_to_trials  : dict {spine_id: [trial_indices]}
        n_spines         : number of unique spines
        time             : number of time points
    """
    ds = xr.load_dataset(cfg["nc_path"])

    # Drop NaN IDs
    valid_mask = ~np.isnan(ds["ids"].values)
    ds = ds.isel(trial=valid_mask)

    raw_traces  = ds["trace"].values.astype(np.float32)  # (n_trials, stim_index, time)
    stim_angles = ds["stim"].values                       # (stim_index,)
    spine_ids   = ds["ids"].values                        # (n_trials,)
    unique_ids  = np.unique(spine_ids)

    n_spines = len(unique_ids)
    time     = raw_traces.shape[2]

    print(f"Unique spines : {n_spines}")
    print(f"Stim positions: {len(stim_angles)}  (12 angles × 8 repeats)")
    print(f"Time points   : {time}")
    print(f"Raw trials    : {len(spine_ids)}  (spine × session combinations)\n")

    # Build lookup: spine_id → list of trial indices for that spine
    spine_to_trials = {sid: np.where(spine_ids == sid)[0] for sid in unique_ids}

    return raw_traces, stim_angles, unique_ids, spine_to_trials, n_spines, time


# ── Augmented dataset ─────────────────────────────────────────────────────────
class MixMatchDataset(Dataset):
    """
    Each sample is a synthetic population vector constructed by independently
    drawing, for each spine, a random session and a random stim-index matching
    the target angle from the training set.
    """
    def __init__(self, raw_traces, stim_angles, unique_ids, spine_to_trials,
                 train_stim_indices, n_samples, n_classes=12):
        self.raw_traces      = raw_traces
        self.stim_angles     = stim_angles
        self.unique_ids      = unique_ids
        self.spine_to_trials = spine_to_trials
        self.n_spines        = len(unique_ids)
        self.time            = raw_traces.shape[2]
        self.n_samples       = n_samples
        self.n_classes       = n_classes

        # For each angle class, which stim_indices (repeats) are in the training set?
        self.class_to_stim_indices = {}
        for cls in range(n_classes):
            angle         = cls * 30
            all_for_angle = np.where(stim_angles == angle)[0]
            available     = np.intersect1d(all_for_angle, train_stim_indices)
            self.class_to_stim_indices[cls] = available

    def __len__(self):
        return self.n_samples

    def __getitem__(self, _):
        # Pick a random angle class
        cls             = np.random.randint(self.n_classes)
        available_stims = self.class_to_stim_indices[cls]

        # Build population vector: (time, n_spines)
        population = np.empty((self.time, self.n_spines), dtype=np.float32)
        for spine_idx, sid in enumerate(self.unique_ids):
            trial    = np.random.choice(self.spine_to_trials[sid])  # random session
            stim_pos = np.random.choice(available_stims)            # random repeat
            population[:, spine_idx] = self.raw_traces[trial, stim_pos, :]

        return torch.tensor(population), torch.tensor(cls, dtype=torch.long)


# ── Real (non-augmented) test set ─────────────────────────────────────────────
class AveragedDataset(Dataset):
    """
    Test set uses session-averaged traces for fair evaluation — no augmentation.
    """
    def __init__(self, raw_traces, stim_angles, unique_ids, spine_to_trials, stim_indices):
        n_spines = len(unique_ids)
        time     = raw_traces.shape[2]
        n        = len(stim_indices)

        X = np.empty((n, time, n_spines), dtype=np.float32)
        for spine_idx, sid in enumerate(unique_ids):
            trial_indices = spine_to_trials[sid]
            avg = raw_traces[trial_indices].mean(axis=0)   # (stim_index, time)
            X[:, :, spine_idx] = avg[stim_indices, :]

        y = (stim_angles[stim_indices] / 30).astype(np.int64)
        self.X = torch.tensor(X)
        self.y = torch.tensor(y)

    def __len__(self):
        return len(self.y)

    def __getitem__(self, i):
        return self.X[i], self.y[i]


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


# ── Main ──────────────────────────────────────────────────────────────────────
def main(cfg=CONFIG):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}\n")

    raw_traces, stim_angles, unique_ids, spine_to_trials, n_spines, time = load_data(cfg)

    # K-fold split over the 96 stim positions (stratified by angle)
    stim_indices = np.arange(len(stim_angles))
    y_stim       = (stim_angles / 30).astype(np.int64)
    skf          = StratifiedKFold(n_splits=cfg["k_folds"], shuffle=True, random_state=42)
    criterion    = nn.CrossEntropyLoss()
    fold_best_accs = []

    for fold, (train_stim_idx, test_stim_idx) in enumerate(skf.split(stim_indices, y_stim), 1):
        print(f"{'─'*60}")
        print(f"Fold {fold}/{cfg['k_folds']}  |  "
              f"train stim positions: {len(train_stim_idx)}  "
              f"test: {len(test_stim_idx)}")
        print(f"{'─'*60}")

        train_ds = MixMatchDataset(
            raw_traces, stim_angles, unique_ids, spine_to_trials,
            train_stim_idx, n_samples=cfg["n_augmented_samples"]
        )
        test_ds = AveragedDataset(
            raw_traces, stim_angles, unique_ids, spine_to_trials, test_stim_idx
        )

        train_loader = DataLoader(train_ds, batch_size=cfg["batch_size"], shuffle=True)
        test_loader  = DataLoader(test_ds,  batch_size=cfg["batch_size"], shuffle=False)

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

        best_acc = 0.0
        for epoch in range(1, cfg["epochs"] + 1):
            train_loss, train_acc = train_epoch(model, train_loader, optimizer, criterion, device)
            test_loss,  test_acc  = eval_epoch(model, test_loader,  criterion, device)
            scheduler.step(test_acc)

            if test_acc > best_acc:
                best_acc = test_acc
                torch.save(model.state_dict(), f"best_rnn_fold{fold}.pt")

            print(
                f"  Epoch {epoch:03d}/{cfg['epochs']} | "
                f"Train loss: {train_loss:.4f}  acc: {train_acc:.3f} | "
                f"Test  loss: {test_loss:.4f}  acc: {test_acc:.3f}"
                + (" ✓" if test_acc == best_acc else "")
            )

        fold_best_accs.append(best_acc)
        print(f"  → Best acc this fold: {best_acc:.3f}\n")

    print(f"{'═'*60}")
    print(f"K-Fold Results ({cfg['k_folds']} folds)")
    print(f"{'═'*60}")
    for i, acc in enumerate(fold_best_accs, 1):
        print(f"  Fold {i}: {acc:.3f}")
    print(f"  Mean ± Std: {np.mean(fold_best_accs):.3f} ± {np.std(fold_best_accs):.3f}")
    print(f"  Chance level: {1/12:.3f}")


if __name__ == "__main__":
    main()