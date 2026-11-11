import csv
import os
from pathlib import Path

data_folder = Path(os.environ["ELLIPTIC_DATA_DIR"])
features_path = data_folder / "elliptic_txs_features.csv"

mapping_path = Path(__file__).parent / "transaction_mapping.csv"
labels_path = Path(__file__).parent / "node_labels.csv"
output_path = Path(__file__).parent / "node_splits.csv"
masks_path = Path(__file__).parent / "node_masks.csv"


def split_data():
    train_ids = []
    val_ids = []
    test_ids = []

    with open(features_path, "r", newline="") as file:
        reader = csv.reader(file)

        for row in reader:
            transaction_id = row[0]
            time_step = int(row[1])

            if time_step <= 34:
                train_ids.append(transaction_id)
            elif time_step <= 41:
                val_ids.append(transaction_id)
            else:
                test_ids.append(transaction_id)

    train_set = set(train_ids)
    val_set = set(val_ids)
    test_set = set(test_ids)

    overlap = (
        len(train_set & val_set)
        + len(train_set & test_set)
        + len(val_set & test_set)
    )

    if overlap > 0:
        raise ValueError("Transactions appear in multiple splits")

    # Verify chronological split
    if len(train_ids) + len(val_ids) + len(test_ids) != 203769:
        raise ValueError("Incorrect total number of transactions")

    print("Training transactions:", len(train_ids))
    print("Validation transactions:", len(val_ids))
    print("Testing transactions:", len(test_ids))

    # Load the transaction mapping
    id_mapping = {}

    with open(mapping_path, "r", newline="") as file:
        reader = csv.reader(file)
        next(reader)

        for row in reader:
            transaction_id = row[0]
            node_id = int(row[1])
            id_mapping[transaction_id] = node_id

    train_nodes = [id_mapping[transaction_id] for transaction_id in train_ids]
    val_nodes = [id_mapping[transaction_id] for transaction_id in val_ids]
    test_nodes = [id_mapping[transaction_id] for transaction_id in test_ids]

    # Save the splits
    with open(output_path, "w", newline="") as file:
        writer = csv.writer(file)
        writer.writerow(["node_id", "split"])

        for node_id in train_nodes:
            writer.writerow([node_id, "train"])

        for node_id in val_nodes:
            writer.writerow([node_id, "validation"])

        for node_id in test_nodes:
            writer.writerow([node_id, "test"])

    # Load transaction labels
    node_labels = {}

    with open(labels_path, "r", newline="") as file:
        reader = csv.reader(file)
        next(reader)

        for row in reader:
            node_id = int(row[0])
            label = row[1]
            node_labels[node_id] = label

    known_labels = 0
    unknown_labels = 0

    for label in node_labels.values():
        if label == "unknown":
            unknown_labels += 1
        else:
            known_labels += 1

    # Create masks
    train_set = set(train_nodes)
    val_set = set(val_nodes)
    test_set = set(test_nodes)

    with open(masks_path, "w", newline="") as file:
        writer = csv.writer(file)
        writer.writerow(["node_id", "train_mask", "val_mask", "test_mask"])

        for node_id in range(len(id_mapping)):
            label = node_labels[node_id]

            train_mask = 0
            val_mask = 0
            test_mask = 0

            if label != "unknown":
                if node_id in train_set:
                    train_mask = 1
                elif node_id in val_set:
                    val_mask = 1
                elif node_id in test_set:
                    test_mask = 1

            writer.writerow([node_id, train_mask, val_mask, test_mask])

    # Check the masks
    train_count = 0
    val_count = 0
    test_count = 0
    invalid_masks = 0

    with open(masks_path, "r", newline="") as file:
        reader = csv.reader(file)
        next(reader)

        for row in reader:
            node_id = int(row[0])
            train_mask = int(row[1])
            val_mask = int(row[2])
            test_mask = int(row[3])

            train_count += train_mask
            val_count += val_mask
            test_count += test_mask

            if train_mask + val_mask + test_mask > 1:
                invalid_masks += 1

            if node_labels[node_id] == "unknown":
                if train_mask + val_mask + test_mask != 0:
                    invalid_masks += 1

    if invalid_masks > 0:
        raise ValueError("Invalid supervised masks found")

    if train_count + val_count + test_count != known_labels:
        raise ValueError("Labeled node count does not match")

    print()
    print("SUPERVISED MASKS")
    print("Training labeled nodes:", train_count)
    print("Validation labeled nodes:", val_count)
    print("Testing labeled nodes:", test_count)
    print()
    print("VALIDATION PASSED")


def main():
    print("CHRONOLOGICAL DATA SPLIT")
    split_data()


if __name__ == "__main__":
    main()
