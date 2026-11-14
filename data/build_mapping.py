import csv
import os
from pathlib import Path


data_folder = Path(os.environ["ELLIPTIC_DATA_DIR"])
output_folder = Path(__file__).parent

features_path = data_folder / "elliptic_txs_features.csv"
classes_path = data_folder / "elliptic_txs_classes.csv"
edges_path = data_folder / "elliptic_txs_edgelist.csv"

mapping_path = output_folder / "transaction_mapping.csv"
labels_path = output_folder / "node_labels.csv"
mapped_edges_path = output_folder / "node_edges.csv"


def build_mapping():
    transaction_ids = []

    with open(features_path, "r", newline="") as file:
        reader = csv.reader(file)

        for row in reader:
            transaction_ids.append((row[0], row[1]))

    id_mapping = {}

    for index, (transaction_id, time_step) in enumerate(transaction_ids):
        id_mapping[transaction_id] = index

    if len(id_mapping) != len(transaction_ids):
        raise ValueError("Duplicate transaction IDs found")

    with open(mapping_path, "w", newline="") as file:
        writer = csv.writer(file)
        writer.writerow(["transaction_id", "node_id", "time_step"])

        for index, (transaction_id, time_step) in enumerate(transaction_ids):
            writer.writerow([transaction_id, index, time_step])

    print("Total mapped transactions:", len(id_mapping))
    print("Mapping saved to:", mapping_path)

    return id_mapping


def map_labels(id_mapping):
    mapped_labels = 0
    unmapped_labels = 0
    node_labels = []

    with open(classes_path, "r", newline="") as file:
        reader = csv.reader(file)
        next(reader)

        for row in reader:
            transaction_id = row[0]
            label = row[1]

            if transaction_id in id_mapping:
                node_id = id_mapping[transaction_id]
                node_labels.append([node_id, label])
                mapped_labels += 1
            else:
                unmapped_labels += 1

    if unmapped_labels > 0 or mapped_labels != len(id_mapping):
        raise ValueError("Some transaction labels could not be mapped")

    node_labels.sort(key=lambda row: row[0])

    with open(labels_path, "w", newline="") as file:
        writer = csv.writer(file)
        writer.writerow(["node_id", "label"])

        for node_id, label in node_labels:
            writer.writerow([node_id, label])

    print("Total mapped labels:", mapped_labels)
    print("Labels saved to:", labels_path)


def map_edges(id_mapping):
    mapped_edges = []
    unmapped_edges = 0

    with open(edges_path, "r", newline="") as file:
        reader = csv.reader(file)
        next(reader)

        for row in reader:
            source_id = row[0]
            target_id = row[1]

            if source_id in id_mapping and target_id in id_mapping:
                source_node = id_mapping[source_id]
                target_node = id_mapping[target_id]

                mapped_edges.append([source_node, target_node])
            else:
                unmapped_edges += 1

    invalid_node_edges = 0

    for source_node, target_node in mapped_edges:
        if source_node < 0 or source_node >= len(id_mapping):
            invalid_node_edges += 1
        elif target_node < 0 or target_node >= len(id_mapping):
            invalid_node_edges += 1

    if unmapped_edges > 0 or invalid_node_edges > 0:
        raise ValueError("Some graph edges could not be mapped correctly")


    with open(mapped_edges_path, "w", newline="") as file:
        writer = csv.writer(file)
        writer.writerow(["source_node", "target_node"])

        for source_node, target_node in mapped_edges:
            writer.writerow([source_node, target_node])

    print("Total mapped edges:", len(mapped_edges))
    print("Edges saved to:", mapped_edges_path)


def main():
    print("TRANSACTION MAPPING")
    id_mapping = build_mapping()

    print("\nLABEL MAPPING")
    map_labels(id_mapping)

    print("\nEDGE MAPPING")
    map_edges(id_mapping)

    print("\nMAPPING COMPLETED SUCCESSFULLY")


if __name__ == "__main__":
    main()
