import csv
import json
import math
import os
from pathlib import Path

data_folder = Path(os.environ["ELLIPTIC_DATA_DIR"])

file_path = data_folder / "elliptic_txs_features.csv"
classes_path = data_folder / "elliptic_txs_classes.csv"
edges_path = data_folder / "elliptic_txs_edgelist.csv"


def validate_features():
    total_transactions = 0
    empty_feature_ids = 0
    invalid_rows = 0
    missing_values = 0
    rows_with_missing_values = 0
    invalid_numbers = 0
    invalid_time_steps = 0

    duplicate_feature_ids = 0
    feature_ids = set()
    time_steps = set()
    transaction_times = {}

    with open(file_path, "r", newline="") as file:
        reader = csv.reader(file)

        for row in reader:
            total_transactions += 1

            if len(row) != 167:
                invalid_rows += 1
                continue

            transaction_id = row[0]

            if transaction_id.strip() == "":
                empty_feature_ids += 1

            if transaction_id in feature_ids:
                duplicate_feature_ids += 1
            else:
                feature_ids.add(transaction_id)

            try:
                time_step = int(row[1])
                transaction_times[transaction_id] = time_step

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

    errors = (
        invalid_rows + duplicate_feature_ids +
        missing_values + invalid_numbers +
        invalid_time_steps + len(set(range(1, 50)) - time_steps) +
        empty_feature_ids)

    print("Total transactions:", total_transactions)
    print("Errors found:", errors)

    return feature_ids, transaction_times, errors


def validate_classes(feature_ids):
    illicit = 0
    licit = 0
    unknown = 0
    total_class_rows = 0
    duplicate_class_ids = 0
    invalid_labels = 0
    invalid_class_rows = 0
    empty_class_ids = 0

    class_ids = set()

    with open(classes_path, "r", newline="") as file:
        reader = csv.reader(file)
        next(reader)  # Skip the header row

        for row in reader:
            total_class_rows += 1

            if len(row) != 2:
                invalid_class_rows += 1
                continue

            transaction_id = row[0]

            if transaction_id.strip() == "":
                empty_class_ids += 1

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
            else:
                invalid_labels += 1

    missing_labels = feature_ids - class_ids
    missing_features = class_ids - feature_ids

    errors = (
        duplicate_class_ids + invalid_labels +
        invalid_class_rows + len(missing_labels) +
        len(missing_features) + empty_class_ids)

    print("Illicit transactions:", illicit)
    print("Licit transactions:", licit)
    print("Unknown transactions:", unknown)
    print("Errors found:", errors)

    return errors


def validate_edges(feature_ids, transaction_times):
    total_edges = 0
    cross_time_edges = 0
    invalid_edges = 0
    duplicate_edges = 0
    self_loops = 0
    invalid_edge_rows = 0
    empty_edge_ids = 0

    edges_seen = set()

    with open(edges_path, "r", newline="") as file:
        reader = csv.reader(file)
        next(reader)  # Skip the header row

        for row in reader:
            total_edges += 1

            if len(row) != 2:
                invalid_edge_rows += 1
                continue

            source_id = row[0]
            target_id = row[1]
            if source_id in transaction_times and target_id in transaction_times:
                if transaction_times[source_id] != transaction_times[target_id]:
                    cross_time_edges += 1

            if source_id.strip() == "" or target_id.strip() == "":
                empty_edge_ids += 1

            if source_id not in feature_ids or target_id not in feature_ids:
                invalid_edges += 1

            edge = (source_id, target_id)

            if edge in edges_seen:
                duplicate_edges += 1
            else:
                edges_seen.add(edge)

            if source_id == target_id:
                self_loops += 1

    errors = (
        invalid_edges + duplicate_edges +
        self_loops + invalid_edge_rows + 
        empty_edge_ids)

    print("Total edges:", total_edges)
    print("Errors found:", errors)

    return errors, cross_time_edges


def main():
    print("FEATURES VALIDATION")
    feature_ids, transaction_times, feature_errors = validate_features()

    print("\nCLASSES VALIDATION")
    class_errors = validate_classes(feature_ids)

    print("\nEDGES VALIDATION")
    edge_errors, cross_time_edges = validate_edges(feature_ids, transaction_times)

    print("Cross-time edges:", cross_time_edges)
    
    edge_errors += cross_time_edges
    
    total_errors = feature_errors + class_errors + edge_errors

    report = {
        "dataset": "Elliptic Bitcoin Dataset",
        "feature_errors": feature_errors,
        "class_errors": class_errors,
        "edge_errors": edge_errors,
        "total_errors": total_errors,
        "status": "PASSED" if total_errors == 0 else "FAILED",
        "temporal_audit": {
            "train_time_steps": "1-34",
            "validation_time_steps": "35-41",
            "test_time_steps": "42-49",
            "cross_time_edges": cross_time_edges,
            "graph_visibility_policy": "past_only_provisional",
            "normalization_policy": "fit_on_training_only_provisional"}}

    report_path = Path(__file__).parent / "validation_report.json"

    with open(report_path, "w") as file:
        json.dump(report, file, indent=4)

    print("\nValidation report saved to:", report_path)

    if total_errors == 0:
        print("\nVALIDATION PASSED")
    else:
        print("\nVALIDATION FAILED")
        print("Total errors:", total_errors)
        raise SystemExit(1)


if __name__ == "__main__":
    main()
