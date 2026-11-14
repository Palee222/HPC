from evaluate import evaluate


def test_evaluation():
    actual = ["1", "2", "1", "2", "1"]
    predicted = ["1", "2", "2", "1", "1"]

    results = evaluate(actual, predicted)

    assert results["tp"] == 2
    assert results["tn"] == 1
    assert results["fp"] == 1
    assert results["fn"] == 1

    assert round(results["precision"], 4) == 0.6667
    assert round(results["recall"], 4) == 0.6667
    assert round(results["f1"], 4) == 0.6667


def test_unknown_labels():
    actual = ["1", "2", "unknown"]
    predicted = ["1", "2", "1"]

    try:
        evaluate(actual, predicted)
        assert False, "Unknown labels should not be accepted"
    except ValueError:
        pass


def test_different_lengths():
    actual = ["1", "2", "1"]
    predicted = ["1", "2"]

    try:
        evaluate(actual, predicted)
    except ValueError:
        return

    assert False, "Different lengths should not be accepted"


def test_no_fraud_predictions():
    actual = ["1", "2", "1"]
    predicted = ["2", "2", "2"]

    results = evaluate(actual, predicted)

    assert results["precision"] == 0
    assert results["recall"] == 0
    assert results["f1"] == 0