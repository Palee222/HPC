def evaluate(y_true, y_pred):
        
    if len(y_true) != len(y_pred):
        raise ValueError("Actual labels and predictions must have the same length")

    for actual, predicted in zip(y_true, y_pred):
        if actual not in ("1", "2"):
            raise ValueError("Actual labels must be 1 or 2")

        if predicted not in ("1", "2"):
            raise ValueError("Predictions must be 1 or 2")

    tp = 0
    tn = 0
    fp = 0
    fn = 0
            
    for actual, predicted in zip(y_true, y_pred):
        if actual == "1" and predicted == "1":
            tp += 1
        elif actual == "2" and predicted == "2":
            tn += 1
        elif actual == "2" and predicted == "1":
            fp += 1
        elif actual == "1" and predicted == "2":
            fn += 1


    precision = 0
    recall = 0
    f1 = 0

    if tp + fp > 0:
        precision = tp / (tp + fp)

    if tp + fn > 0:
        recall = tp / (tp + fn)

    if precision + recall > 0:
        f1 = 2 * precision * recall / (precision + recall)


    return {
        "tp": tp,
        "tn": tn,
        "fp": fp,
        "fn": fn,
        "precision": precision,
        "recall": recall,
        "f1": f1
    }

def main():
    actual = ["1", "2", "1", "2", "1"]
    predicted = ["1", "2", "2", "1", "1"]

    results = evaluate(actual, predicted)
    print("Precision:", round(results["precision"], 4))
    print("Recall:", round(results["recall"], 4))
    print("F1-score:", round(results["f1"], 4))


if __name__ == "__main__":
    main()