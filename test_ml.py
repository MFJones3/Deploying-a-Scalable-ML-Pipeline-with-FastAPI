import numpy as np

from ml.model import (
    inference,
    compute_model_metrics,
    save_model,
    load_model,
)

from sklearn.ensemble import RandomForestClassifier


def test_inference_returns_one_prediction_per_record():
    """Inference should return one prediction for each input row."""
    X = np.array([
        [20, 1],
        [35, 0],
        [45, 1],
        [60, 0],
        [25, 1],
        [55, 0],
    ])
    y = np.array([0, 0, 1, 1, 0, 1])

    model = RandomForestClassifier(n_estimators=10, random_state=42)
    model.fit(X, y)

    predictions = inference(model, X)

    assert len(predictions) == len(X)
    assert set(predictions).issubset({0, 1})


def test_model_metrics_are_correct():
    """Metrics should match known classification results."""
    y_true = np.array([1, 1, 0, 0])
    y_pred = np.array([1, 0, 0, 0])

    precision, recall, f1 = compute_model_metrics(y_true, y_pred)

    assert np.isclose(precision, 1.0)
    assert np.isclose(recall, 0.5)
    assert np.isclose(f1, 2 / 3)


def test_model_save_and_load(tmp_path):
    """A saved and reloaded model should produce identical predictions."""
    X = np.array([[20, 1], [35, 0], [45, 1], [60, 0]])
    y = np.array([0, 0, 1, 1])

    model = RandomForestClassifier(n_estimators=10, random_state=42)
    model.fit(X, y)

    model_path = tmp_path / "test_model.pkl"

    save_model(model, str(model_path))
    loaded_model = load_model(str(model_path))

    assert model_path.exists()

    original_predictions = inference(model, X)
    loaded_predictions = inference(loaded_model, X)

    assert np.array_equal(original_predictions, loaded_predictions)
