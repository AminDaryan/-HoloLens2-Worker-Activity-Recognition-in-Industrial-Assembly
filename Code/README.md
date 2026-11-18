# Code: gaze features and activity classifiers

This folder contains Python code that classifies the four activities of this dataset from
the gaze recordings. It cuts each recording into 240-row windows
(8 s at the headset's nominal 30 Hz), computes the 19 gaze features of GEAR [1, 2] for every
window, trains the feature-based classifiers of GEAR (support vector machines, random forest,
extra trees) and a small LSTM network on them, and scores their predictions.

What this code is not:

- It does not contain the four models reported in the paper (the CNN-LSTM, the Transformer,
  and their fused versions) or their trained weights.
- Its 19 features are not the expert-feature vector of the paper's fused models. That
  vector has 24 dimensions and also includes gaze-motion, gaze-origin, and gaze-validity
  measures; the code that computed it is not part of this repository.

The results of these scripts are therefore not results of the paper.

---

## Folders

The code is split into the steps of the analysis. Preprocessing and the gaze plotting script in Postprocessing read the
recordings in `Participants Gaze Dataset/`. The classifiers read `Code/output/features.csv`,
and the evaluation reads the `predictions_*.csv` files there. Every script writes its files to
`Code/output/`.

| Folder | File | What it does |
|---|---|---|
| `Preprocessing/` | `gaze_data.py` | Reads `recordings.csv` and the recordings, and cuts the recordings into windows with their training, validation, or test label |
| | `gear_features.py` | The 19 GEAR features of one window, including the fixation detection |
| | `extract_features.py` | Computes the features of all 338 windows → `output/features.csv` |
| `Classification Algorithms/` | `feature_classifiers.py` | Trains and tests the SVM, random-forest, and extra-trees classifiers → `output/predictions_temporal.csv` or `output/predictions_participant.csv` |
| | `lstm_classifier.py` | Trains and tests the LSTM network (needs PyTorch) → `output/predictions_lstm.csv` |
| `Postprocessing/` | `evaluate_predictions.py` | Scores predictions and draws confusion matrices → `output/results_*.csv`, `output/confusion_*.csv`, `output/confusion_*.png` |
| | `plot_gaze_directions.py` | 3D plot of the gaze directions of the four recordings in one participant folder |
| `tests/` | | Checks of every step |

`requirements.txt` and `requirements-lstm.txt` list the Python packages needed, and
`LICENSE` holds the Apache License 2.0, which covers the code in this folder.

---

## Quick start

You need Python 3.10 or newer. Run all commands from the root of the repository (the folder
that contains `Participants Gaze Dataset` and `Code`). The quotation marks are needed because
`Classification Algorithms` contains a space.

On Windows, replace the first two lines with `py -m venv .venv` and `.venv\Scripts\activate`.
If PowerShell refuses to run the activation script, run
`Set-ExecutionPolicy -Scope CurrentUser RemoteSigned` once. On Linux, run
`pip install torch --index-url https://download.pytorch.org/whl/cpu` before
`pip install -r Code/requirements-lstm.txt`: the default PyTorch build from PyPI adds about
4 GB of GPU libraries that the LSTM does not use.

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r Code/requirements.txt

python Code/Preprocessing/extract_features.py
python "Code/Classification Algorithms/feature_classifiers.py"
python "Code/Classification Algorithms/feature_classifiers.py" --protocol participant
python Code/Postprocessing/evaluate_predictions.py
python Code/Postprocessing/plot_gaze_directions.py --participant P03

pip install -r Code/requirements-lstm.txt
python "Code/Classification Algorithms/lstm_classifier.py"
python Code/Postprocessing/evaluate_predictions.py

pip install pytest
python -m pytest Code/tests
```

The first `feature_classifiers.py` command uses the temporal holdout of the paper, and
`--protocol participant` leaves one participant out (see
[Evaluation protocols](#evaluation-protocols)). The LSTM script trains the network five times.
Every script prints a summary and writes its files to `Code/output/`. Feature extraction and
the feature classifiers take a few seconds; the five LSTM runs take about a minute on a laptop
CPU. `--help` lists the options of each script.

---

## Windows and data splits

The current data files are cut into windows as follows:

- Each recording is cut into consecutive, non-overlapping windows of 240 rows (8 s at the
  headset's nominal 30 Hz), starting at its first row. An incomplete last window is dropped.
- The windows of each recording are split in time order: the first 70 % (rounded down) for
  training, the next 15 % (rounded down) for validation, and the rest for testing.

This gives 338 windows: 217 for training, 39 for validation, and 82 for testing. The test set has 21 assembling, 20 gluing, 21 waiting, and 20 walking windows.

Rates are divided by each window's measured duration. The columns `n_samples` and `duration_s` of `output/features.csv` give the size of every window.

---

## Features

The features are computed for each window. Fixations are detected with the
dispersion-threshold (I-DT) method of GEAR on the samples with a valid gaze direction and a
valid calibration: a fixation lasts at least 100 ms, and its gaze directions stay within a
dispersion of 1.6°.

| Features | Meaning |
|---|---|
| `meanFix`, `minFix`, `maxFix`, `varFix`, `stdFix` | Mean, minimum, maximum, variance, and standard deviation of the fixation durations (ms) |
| `meanDis`, `minDis`, `maxDis`, `varDis`, `stdDisp` | The same statistics of the fixation dispersions (radians) |
| `freqDisPerSec` | Number of fixations per second |
| `number_of_blinks` | Number of runs of consecutive samples without a gaze value. Short losses of tracking are counted as well, because the headset reports them in the same way |
| `blinkMean`, `blinkMin`, `blinkMax` | Mean, minimum, and maximum duration of these runs (ms) |
| `blinkRate` | Number of such runs per second |
| `xDir`, `yDir` | Share of successive fixations whose center moves in the positive x or y direction; −1 with fewer than two fixations |
| `fixDensPerBB` | Number of fixations divided by the area of the bounding box of the x and y gaze-direction components; −1 when the area is zero |

Statistics that are undefined for a window (for example the variance with fewer than two
fixations) are set to 0. The feature names are those of the original GEAR code.

---

## Classifiers

| Model | Settings |
|---|---|
| SVM, linear kernel | C = 5000 |
| SVM, polynomial kernel | degree 3, C = 5000 |
| SVM, RBF kernel | C = 2000, gamma = `scale` |
| SVM, sigmoid kernel | C = 1 |
| Random forest | 100 trees, random state 96 |
| Extra trees | 38 trees, random state 63 |
| LSTM | 2 layers of 128 units, dropout 0.3, Adam (learning rate 0.00894, weight decay 3.249 × 10⁻⁶), batch size 16, at most 500 epochs, early stopping after 50 epochs without a better validation accuracy |

For the SVMs, the random forest, and the LSTM, the features are scaled by their maximum
absolute value, fitted on the training windows only. Extra trees are trained on the unscaled
features, as in the original notebook. The LSTM network reads the 19 features of one window
as a sequence of length one, so it does not use the order of the samples or of the windows.
It is trained five times (random seeds 0 to 4), and every run is reported.

These settings are not those of upstream GEAR. They come from later versions of the
notebooks, which were run on the same recordings with 10-second windows and random splits.
Those notebooks contain searches over some of the settings (the random seeds of both
forests, the number of extra trees, and C for one SVM kernel) that keep the best accuracy on
the held-out part of a random split. None of the settings were tuned again for the splits
used here, but the test windows of both protocols below are not independent of those
searches, so the settings should not be read as tuned values.

---

## Evaluation protocols

`feature_classifiers.py` offers two protocols:

- **Temporal holdout** (`--protocol temporal`, the default; also used by
  `lstm_classifier.py`). This is the protocol of the paper, which calls it subject-inclusive
  evaluation with non-overlapping temporal holdout. The models are trained on the training
  windows and tested on the test windows; the feature classifiers also predict the
  validation windows, and the LSTM uses them for early stopping. Every participant appears
  in all three parts, so this protocol measures how well later parts of recordings that were
  seen in training are recognized. It does not measure how well the models work for people
  they were not trained on.
- **Leave one participant out** (`--protocol participant`). All windows of one participant
  are held out, the models are trained on the other nine participants, and this is repeated
  for each participant. The predictions of the ten rounds are pooled. Participants are
  grouped by the `recorded_by` column of `recordings.csv`, so `P04/P04_Waiting.csv` is held
  out with P10 and `P05/P05_Walking.csv` with P09. The paper does not report this protocol.

Consecutive windows of one recording are not independent, and most participants have only
one recording of each activity (P10 has two waiting and P09 two walking recordings, while
P04 has no waiting and P05 no walking recording of their own). With 82 test windows, one
window more or less changes the accuracy by about 1.2 percentage points. Differences of a few
windows between models should therefore not be read as real differences.

---

## Output files

All files are written to `Code/output/`, which git ignores.

| File | Written by | Content |
|---|---|---|
| `features.csv` | `extract_features.py` | One row per window: recording, participant folder (`slot`), recorded participant (`recorded_by`), activity, window number, split, row range in the file, number of samples, number of valid samples, duration in seconds, and the 19 features |
| `predictions_temporal.csv`, `predictions_participant.csv` | `feature_classifiers.py` | One row per model and evaluated window: the window's recording, number, recorded participant and activity, its split (temporal holdout only), and the predicted activity |
| `predictions_lstm.csv`, `lstm_runs.csv` | `lstm_classifier.py` | The same for every LSTM run on the test windows, and the best validation epoch and accuracy of each run |
| `results_<name>.csv` | `evaluate_predictions.py` | Number of windows, accuracy, macro-averaged F1 score, and the recall of each activity, for every model (and every split and run) |
| `confusion_<name>.csv`, `confusion_<name>.png` | `evaluate_predictions.py` | Confusion matrices on the test windows; for the LSTM summed over the runs, and for leave one participant out over all windows |
| `gaze_directions_<participant>.png` | `plot_gaze_directions.py` | The 3D gaze plots; with `--window NN` the name ends in `_wNN.png` |

The feature classifiers give the same results on every run, and they gave the same results
with the minimum package versions of `requirements.txt` on Python 3.10 and with the newest
versions as of November 2026 on Python 3.13. The LSTM results repeat on the same computer but
can differ slightly between PyTorch versions and computers.

---

## Changes from the original GEAR code

The feature functions and classifiers were taken from the GEAR2 notebooks of GEAR
(`FeatureCalculation.ipynb`, `AnSVMClassifierForHL2GazeFeatures.ipynb`,
`RandomForestClassifier.ipynb`, `ExtraTreesClassifier.ipynb`) and rewritten as Python
scripts. The LSTM network and the 3D plot come from later notebooks written for this dataset.
The main changes are:

- **Windows and splits.** The notebooks cut recordings into windows of a fixed duration and
  split the windows at random. Here the 240-row windows and the time-ordered splits of the
  paper are used, and the feature scaling is fitted on the training windows only.
- **Classifier settings.** The settings are those of later versions of the notebooks
  written for this dataset, not those of upstream GEAR (see [Classifiers](#classifiers)).
- **Fixation dispersion.** For a fixation that ended because the next sample exceeded the
  threshold, the original reported the dispersion including that next sample, so most
  reported dispersions were above the 1.6° threshold. The dispersion of the fixation's own
  samples is reported here.
- **Blinks.** The original counted every sample without a gaze value as a separate blink.
  It gave each one a duration equal to the time since a reference point that moved only
  after more than one second had passed, so durations of several seconds were reported.
  Here each run of consecutive invalid samples is one blink, and its duration is measured
  from its first sample to the next sample.
- **Fixation density.** The original used the x extent of the gaze directions for both sides
  of the bounding box. The y extent is used for the height here.
- **Rates.** Fixations and blinks per second are divided by the measured duration of the
  window.
- **Undefined values and invalid input.** Statistics that the original left undefined are
  set to 0 (see [Features](#features)). Samples whose gaze direction is missing or of length
  zero are not used for fixations, and the arc cosine in the dispersion is protected against
  rounding errors; neither changes any value on this dataset.
- **LSTM.** The notebook kept a reference to the weights of the best epoch, not a copy, so
  the weights of the last epoch were used. It also showed the confusion matrix of the run
  with the highest test accuracy. Here the weights of the best validation epoch are copied,
  and every run is reported.
- **Evaluation.** The classifiers write their prediction for every evaluated window (the
  validation and test windows of the temporal holdout, all windows for leave one participant
  out, and the test windows of every LSTM run), and `evaluate_predictions.py` computes the
  scores and confusion matrices from them.

The fixation detection algorithm and its thresholds are unchanged. The Unity application,
the online feature server, data augmentation, and hyperparameter searches of the original
notebooks are not included.

---

## License and attribution

The code in this folder is licensed under the Apache License 2.0; the full text is in
[`LICENSE`](LICENSE). The recordings and their documentation in the rest of the repository
remain under CC BY 4.0 (see the main [README](../README.md#license-and-consent)).

The code is adapted from the GEAR2 notebooks of GEAR
(<https://github.com/Interactions-HSG/GEAR>), by the Interaction- and Communication-based
Systems group at the University of St.Gallen, which are licensed under the Apache License
2.0 and are supplementary material of [1] and [2]. Their fixation detection is in turn
adapted from the Pupil Labs tutorials (<https://github.com/pupil-labs/pupil-tutorials>),
and their SVM set-up is a modified version of the classifier described in
[this article](https://towardsdatascience.com/multiclass-classification-with-support-vector-machines-svm-kernel-trick-kernel-functions-f9d5377d6f02).
Each file states the changes made to the original code. If you use the features, please
also cite [1] and [2].

---

## References

[1] K. Bektaş, J. Strecker, S. Mayer, K. Garcia, J. Hermann, K. E. Jenss, Y. S. Antille,
and M. E. Solèr, "GEAR: Gaze-enabled augmented reality for human activity recognition," in
*Proc. 2023 Symposium on Eye Tracking Research and Applications (ETRA '23)*, Tübingen,
Germany, 2023, doi: [10.1145/3588015.3588402](https://doi.org/10.1145/3588015.3588402).

[2] K. Bektaş, J. Strecker, S. Mayer, and K. Garcia, "Gaze-enabled activity recognition for
augmented reality feedback," *Computers & Graphics*, vol. 119, art. no. 103909, 2024,
doi: [10.1016/j.cag.2024.103909](https://doi.org/10.1016/j.cag.2024.103909).
