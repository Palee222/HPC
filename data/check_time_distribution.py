import csv
import os
from pathlib import Path

data_folder = Path(os.environ["ELLIPTIC_DATA_DIR"])

features_path = data_folder / "elliptic_txs_features.csv"
classes_path = data_folder / "elliptic_txs_classes.csv"

transaction_times = {}
time_counts = {}

with open(features_path, "r", newline="") as file:
    reader = csv.reader(file)

    for row in reader:
        transaction_id = row[0]
        time_step = int(row[1])

        transaction_times[transaction_id] = time_step
        if time_step not in time_counts:
            time_counts[time_step] = {"1": 0, "2": 0, "unknown": 0}

with open(classes_path, "r", newline="") as file:
    reader = csv.reader(file)
    next(reader)

    for row in reader:
        transaction_id = row[0]
        label = row[1]

        time_step = transaction_times[transaction_id]
        time_counts[time_step][label] += 1

print("Step | Illicit | Licit | Unknown")
print("--------------------------------")

for step in sorted(time_counts):
    counts = time_counts[step]
    print(step, "|", counts["1"], "|", counts["2"], "|", counts["unknown"])
