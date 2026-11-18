"""Classify the four activities with a small LSTM network on the 19 gaze
features of each window.

Usage, from the repository root, after Code/Preprocessing/extract_features.py
(needs PyTorch: pip install -r Code/requirements-lstm.txt):

    python "Code/Classification Algorithms/lstm_classifier.py"

Trains the network five times (random seeds 0 to 4) on the training windows of
the paper's temporal split, stops each run early on the validation windows, and
writes the predictions for the test windows to Code/output/predictions_lstm.csv
and the best epoch of each run to Code/output/lstm_runs.csv.

The network reads the 19 features of one window as a sequence of length one, so
it does not model the order of the samples or of the windows. Its settings, the
constants below, are those of the notebook it is adapted from (LTSM NN
PyTorch.ipynb, written for this dataset and run with random splits; the settings
were not tuned again here).

Changes made in 2026:

- The windows are split by the paper's temporal protocol instead of at random,
  and the feature scaling is fitted on the training windows only.
- The weights of the best validation epoch are copied. The notebook saved them
  with state_dict().copy(), which keeps references to tensors that later
  training steps overwrite, so it used the weights of the last epoch.
- Every run is reported. The notebook restored the run with the highest test
  accuracy and showed only its confusion matrix.

Licensed under the Apache License 2.0 (see Code/LICENSE).
"""

from __future__ import annotations

import argparse
import copy
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.preprocessing import MaxAbsScaler

# The code folders are not Python packages, so the shared modules are imported by path.
CODE_DIR = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(CODE_DIR / "Preprocessing"), str(CODE_DIR / "Postprocessing")]
from evaluate_predictions import print_scores, results_table  # noqa: E402
from extract_features import FEATURES_FILE, WINDOW_KEYS, read_features  # noqa: E402
from gaze_data import ACTIVITIES, OUTPUT_DIR, SPLITS  # noqa: E402
from gear_features import FEATURE_NAMES  # noqa: E402


def make_parser() -> argparse.ArgumentParser:
    """The command-line options. They are defined before PyTorch is imported, so
    that --help works without it."""
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--features", type=Path, default=FEATURES_FILE,
                        help="feature table (default: Code/output/features.csv)")
    parser.add_argument("--runs", type=int, default=5,
                        help="number of training runs, with random seeds 0, 1, ... (default: 5)")
    parser.add_argument("--out-dir", type=Path, default=OUTPUT_DIR,
                        help="folder for the predictions (default: Code/output)")
    return parser


try:
    import torch
    from torch import nn
    from torch.utils.data import DataLoader, TensorDataset
except ImportError:
    if __name__ == "__main__":
        make_parser().parse_args()  # answers --help and rejects unknown options
    raise SystemExit("PyTorch is not installed; run: pip install -r Code/requirements-lstm.txt")

HIDDEN_SIZE = 128
NUM_LAYERS = 2
DROPOUT = 0.3
LEARNING_RATE = 0.00894
WEIGHT_DECAY = 3.249e-06
BATCH_SIZE = 16
MAX_EPOCHS = 500
PATIENCE = 50


class LSTMClassifier(nn.Module):
    """LSTM layers, dropout, and a linear layer with one output per activity."""

    def __init__(self, input_size: int, num_classes: int):
        super().__init__()
        self.lstm = nn.LSTM(input_size, HIDDEN_SIZE, NUM_LAYERS, batch_first=True)
        self.dropout = nn.Dropout(DROPOUT)
        self.fc = nn.Linear(HIDDEN_SIZE, num_classes)

    def forward(self, x):
        out, _ = self.lstm(x.unsqueeze(1))  # one time step per window
        return self.fc(self.dropout(out[:, -1, :]))


def to_tensors(features: pd.DataFrame) -> dict[str, tuple]:
    """Features and labels of the training, validation and test windows as
    tensors. The scaling is fitted on the training windows."""
    parts = {split: features[features["split"] == split] for split in SPLITS}
    scaler = MaxAbsScaler().fit(parts["train"][FEATURE_NAMES])
    label_of = {activity: i for i, activity in enumerate(ACTIVITIES)}
    return {split: (torch.tensor(scaler.transform(part[FEATURE_NAMES]), dtype=torch.float32),
                    torch.tensor(part["activity"].map(label_of).to_numpy(), dtype=torch.long))
            for split, part in parts.items()}


def predict(model, x) -> np.ndarray:
    """Index of the predicted activity for every row of x."""
    model.eval()
    with torch.no_grad():
        return model(x).argmax(dim=1).numpy()


def train_one_run(seed: int, x_train, y_train, x_val, y_val):
    """Train one network and return it with the weights of its best validation
    epoch, that epoch's validation accuracy, and the epoch number."""
    torch.manual_seed(seed)
    np.random.seed(seed)
    model = LSTMClassifier(x_train.shape[1], len(ACTIVITIES))
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
    loss_fn = nn.CrossEntropyLoss()
    loader = DataLoader(TensorDataset(x_train, y_train), batch_size=BATCH_SIZE, shuffle=True,
                        generator=torch.Generator().manual_seed(seed))
    best_accuracy, best_state, best_epoch = -1.0, None, 0
    for epoch in range(1, MAX_EPOCHS + 1):
        model.train()
        for inputs, labels in loader:
            optimizer.zero_grad()
            loss_fn(model(inputs), labels).backward()
            optimizer.step()
        accuracy = float((predict(model, x_val) == y_val.numpy()).mean())
        if accuracy > best_accuracy:
            best_accuracy, best_state, best_epoch = accuracy, copy.deepcopy(model.state_dict()), epoch
        elif epoch - best_epoch >= PATIENCE:
            break
    model.load_state_dict(best_state)
    return model, best_accuracy, best_epoch


def predict_runs(features: pd.DataFrame, n_runs: int) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Train one network for each random seed 0, 1, ... and predict the test windows.

    Returns the predictions of all runs, and the best epoch and its validation
    accuracy for every run.
    """
    data = to_tensors(features)
    test = features[features["split"] == "test"]
    tables, runs = [], []
    for seed in range(n_runs):
        model, val_accuracy, best_epoch = train_one_run(seed, *data["train"], *data["val"])
        table = test[WINDOW_KEYS + ["split"]].copy()
        table.insert(0, "model", "LSTM")
        table.insert(1, "run", seed)
        table["predicted"] = [ACTIVITIES[i] for i in predict(model, data["test"][0])]
        tables.append(table)
        runs.append({"run": seed, "best_epoch": best_epoch, "val_accuracy": val_accuracy})
        print(f"run {seed}: best validation accuracy {val_accuracy:.3f} in epoch {best_epoch}")
    return pd.concat(tables, ignore_index=True), pd.DataFrame(runs)


def main(argv: list[str] | None = None) -> int:
    parser = make_parser()
    args = parser.parse_args(argv)
    if args.runs < 1:
        parser.error("--runs must be at least 1")
    try:
        features = read_features(args.features)
    except ValueError as problem:
        parser.error(str(problem))
    if not (features["split"] == "val").any():
        parser.error(f"{args.features} has no validation windows, which early stopping needs")
    torch.set_num_threads(1)  # a small network; one thread keeps the runs repeatable

    predictions, runs = predict_runs(features, args.runs)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    out = args.out_dir / "predictions_lstm.csv"
    predictions.to_csv(out, index=False)
    runs.to_csv(args.out_dir / "lstm_runs.csv", index=False)
    print()
    print_scores(results_table(predictions))
    print(f"\nPredictions written to {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
