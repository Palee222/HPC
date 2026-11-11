# Elliptic Dataset — Data and Evaluation

## Dataset Setup

Download the Elliptic Bitcoin Dataset and extract these three files:

- `elliptic_txs_features.csv`
- `elliptic_txs_classes.csv`
- `elliptic_txs_edgelist.csv`

Keep the original dataset outside the GitHub repository.

Set the dataset location in PowerShell:

```powershell
$env:ELLIPTIC_DATA_DIR = "C:\path\to\elliptic_bitcoin_dataset"
```

Replace the example path with the actual folder containing the three CSV files.

## Generate the Data

From the repository root, run these commands in order:

```powershell
python data/check_checksums.py
python data/validate_data.py
python data/build_mapping.py
python data/split_data.py
python -m pytest data/test_evaluate.py -v
```

These commands validate the original dataset, generate the node mapping and chronological splits, and test the evaluation function.


## Generated Files

- `transaction_mapping.csv` — Maps original transaction IDs to canonical node IDs (0–203768) and includes each transaction's time step (1–49).
- `node_labels.csv` — Stores each node's label: 1 = illicit, 2 = licit, unknown = unlabeled.
- `node_edges.csv` — Stores connections between nodes using canonical node IDs.
- `node_splits.csv` — Assigns every node to training, validation or testing.
- `node_masks.csv` — Identifies labeled nodes used for supervised training and evaluation.
- `validation_report.json` — Records dataset validation results and the provisional temporal leakage policy.
- `time_distribution.txt` — Shows illicit, licit, and unknown transaction counts for all 49 time steps.

## Time Distribution Audit

To regenerate the time-step counts, run:

python data/check_time_distribution.py

The report covers all 49 time steps and totals 203,769 transactions:
- 4,545 illicit
- 42,019 licit
- 157,205 unknown

The saved results are in `data/time_distribution.txt`.

All node IDs refer to the same transactions across these files.

## Chronological Splits

- Training: time steps 1–34
- Validation: time steps 35–41
- Testing: time steps 42–49

Unknown labels are excluded from the supervised training and evaluation.

## Temporal Leakage Policy

- Predictions must not use information from future time steps.
- All 234,355 dataset edges connect transactions within the same time step.
- Unknown labels must not be used as supervised targets or input features.
- Feature normalization should be fitted using training data only.
- Validation data is used for model selection; test data must not be used for tuning.

These rules must also be followed by the graph algorithms and GNN implementations.

## Data Loading Contract

The original `elliptic_txs_features.csv` contains no header.

Each row contains:
- Column 0: Original transaction ID
- Column 1: Time step
- Columns 2–166: 165 numerical features

The canonical `node_id` is the zero-based row index in the original features file.

All generated files use these canonical node IDs.

When combining labels, edges, splits, and masks with features, match them using `node_id`, not CSV row position.

The positive class for evaluation is `"1"` (illicit).
The negative class is `"2"` (licit).
Unknown labels must be excluded from supervised evaluation.

## Evaluation Function

The shared evaluation function is in `data/evaluate.py`.

It calculates:
- True positives (TP)
- True negatives (TN)
- False positives (FP)
- False negatives (FN)
- Precision, recall, and F1-score

Class `"1"` is illicit (positive).
Class `"2"` is licit (negative).

Only labeled validation or test nodes should be passed to the function. Unknown labels must be excluded.

Example:

```python
from data.evaluate import evaluate

actual = ["1", "2", "1", "2"]
predicted = ["1", "2", "2", "1"]

results = evaluate(actual, predicted)
print(results)
```

Run the evaluation tests from the repository root:

```powershell
python -m pytest data/test_evaluate.py -v
```