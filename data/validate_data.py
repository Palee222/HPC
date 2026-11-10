
import csv
import math
import os
from pathlib import Path

data_folder = Path(os.environ["ELLIPTIC_DATA_DIR"])

file_path = data_folder / "elliptic_txs_features.csv"
classes_path = data_folder / "elliptic_txs_classes.csv"
edges_path = data_folder / "elliptic_txs_edgelist.csv"


def validate_features():
    total_transactions = 0
    invalid_rows = 0
    missing_values = 0
    rows_with_missing_values = 0
    invalid_numbers = 0
    invalid_time_steps = 0

    feature_ids = set()
    time_steps = set()

    with open(file_path, "r", newline="") as file:
        reader = csv.reader(file)

        for row in reader:
            total_transactions += 1

            if len(row) != 167:
                invalid_rows += 1
                continue

            transaction_id = row[0]
            feature_ids.add(transaction_id)

            try:
                time_step = int(row[1])

                if time_step < 1 or time_step > 49:
                    invalid_time_steps += 1
                else:
                    time_steps.add(time_step)

            except ValueError:
                invalid_time_steps += 1

            features = row[2:]
            missing_in_row = 0

            for value in features:
                if value.strip() == "":
                    missing_values += 1
                    missing_in_row += 1

                try:
                    number = float(value)

                    if not math.isfinite(number):
                        invalid_numbers += 1

                except ValueError:
                    invalid_numbers += 1

            if missing_in_row > 0:
                rows_with_missing_values += 1

    print("Total transactions:", total_transactions)
    print("Rows with incorrect column count:", invalid_rows)
    print("Unique feature IDs:", len(feature_ids))
    print("Total missing feature values:", missing_values)
    print("Transactions with missing features:", rows_with_missing_values)
    print("Invalid numerical feature values:", invalid_numbers)
    print("Number of unique time steps:", len(time_steps))
    print("Earliest time step:", min(time_steps))
    print("Latest time step:", max(time_steps))
    print("Invalid time steps:", invalid_time_steps)
    print("Missing time steps:", sorted(set(range(1, 50)) - time_steps))

    return feature_ids


def validate_classes(feature_ids):
    illicit = 0
    licit = 0
    unknown = 0
    total_class_rows = 0
    duplicate_class_ids = 0

    class_ids = set()

    with open(classes_path, "r", newline="") as file:
        reader = csv.reader(file)
        next(reader)  # Skip the header row

        for row in reader:
            total_class_rows += 1

            transaction_id = row[0]
            label = row[1]

            if transaction_id in class_ids:
                duplicate_class_ids += 1
            else:
                class_ids.add(transaction_id)

            if label == "1":
                illicit += 1
            elif label == "2":
                licit += 1
            elif label == "unknown":
                unknown += 1

    missing_labels = feature_ids - class_ids
    missing_features = class_ids - feature_ids

    print("Illicit transactions:", illicit)
    print("Licit transactions:", licit)
    print("Unknown transactions:", unknown)
    print("Total class rows:", total_class_rows)
    print("Unique class IDs:", len(class_ids))
    print("Duplicate class IDs:", duplicate_class_ids)
    print("Transactions missing labels:", len(missing_labels))
    print("Labels without matching features:", len(missing_features))


def validate_edges(feature_ids):
    total_edges = 0
    invalid_edges = 0
    duplicate_edges = 0
    self_loops = 0

    edges_seen = set()

    with open(edges_path, "r", newline="") as file:
        reader = csv.reader(file)
        next(reader)  # Skip the header row

        for row in reader:
            total_edges += 1

            source_id = row[0]
            target_id = row[1]

            if source_id not in feature_ids or target_id not in feature_ids:
                invalid_edges += 1

            edge = (source_id, target_id)

            if edge in edges_seen:
                duplicate_edges += 1
            else:
                edges_seen.add(edge)

            if source_id == target_id:
                self_loops += 1

    print("Total edges:", total_edges)
    print("Edges with missing transaction IDs:", invalid_edges)
    print("Unique directed edges:", len(edges_seen))
    print("Duplicate directed edges:", duplicate_edges)
    print("Self-loops:", self_loops)


def main():
    print("FEATURES VALIDATION")
    feature_ids = validate_features()

    print("\nCLASSES VALIDATION")
    validate_classes(feature_ids)

    print("\nEDGES VALIDATION")
    validate_edges(feature_ids)


if __name__ == "__main__":
    main()
