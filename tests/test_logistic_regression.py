import numpy as np
import pytest
from src.logistic_regression import LogisticRegression


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def binary_data():
    """Linearly separable 2-class dataset."""
    rng = np.random.default_rng(42)
    X0 = rng.normal(loc=-2.0, scale=0.8, size=(50, 2))
    X1 = rng.normal(loc=2.0, scale=0.8, size=(50, 2))
    X = np.vstack([X0, X1])
    y = np.array([0] * 50 + [1] * 50)
    return X, y


@pytest.fixture
def multiclass_data():
    """Linearly separable 3-class dataset."""
    rng = np.random.default_rng(42)
    X0 = rng.normal(loc=[-3.0, 0.0], scale=0.6, size=(40, 2))
    X1 = rng.normal(loc=[3.0, 0.0], scale=0.6, size=(40, 2))
    X2 = rng.normal(loc=[0.0, 3.5], scale=0.6, size=(40, 2))
    X = np.vstack([X0, X1, X2])
    y = np.array([0] * 40 + [1] * 40 + [2] * 40)
    return X, y


# ---------------------------------------------------------------------------
# Unit Tests: Activation & Loss Functions
# ---------------------------------------------------------------------------

def test_sigmoid_values_and_clipping():
    model = LogisticRegression()
    scores = np.array([-1000.0, 0.0, 1000.0])
    out = model.sigmoid(scores)

    assert np.isclose(out[1], 0.5)
    assert np.all(np.isfinite(out))
    assert np.all((out >= 0.0) & (out <= 1.0))


def test_softmax_rows_sum_to_one():
    model = LogisticRegression()
    scores = np.array([
        [1.0, 2.0, 3.0],
        [-1000.0, 0.0, 1000.0],
    ])
    probs = model.softmax(scores)

    assert probs.shape == (2, 3)
    assert np.all(np.isfinite(probs))
    assert np.allclose(np.sum(probs, axis=1), np.ones(2))


def test_compute_loss_finite_on_extreme_predictions():
    model = LogisticRegression()
    y = np.array([[1.0], [0.0]])
    y_hat = np.array([[0.0], [1.0]])  # Worst possible predictions
    loss = model.compute_loss(y, y_hat)

    assert isinstance(loss, float)
    assert np.isfinite(loss)
    assert loss > 0.0


# ---------------------------------------------------------------------------
# Error Handling & Edge Cases
# ---------------------------------------------------------------------------

def test_predict_proba_before_fit_raises():
    model = LogisticRegression()
    X = np.array([[1.0, 2.0], [3.0, 4.0]])
    with pytest.raises(RuntimeError):
        model.predict_proba(X)


def test_predict_before_fit_raises():
    model = LogisticRegression()
    X = np.array([[1.0, 2.0], [3.0, 4.0]])
    with pytest.raises(RuntimeError):
        model.predict(X)


def test_single_class_raises_value_error():
    model = LogisticRegression()
    X = np.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])
    y = np.array([1, 1, 1])
    with pytest.raises(ValueError):
        model.fit(X, y)


def test_reproducibility_with_random_state(binary_data):
    X, y = binary_data
    m1 = LogisticRegression(lr=0.1, epoch=30, batch_size=16, random_state=123).fit(X, y)
    m2 = LogisticRegression(lr=0.1, epoch=30, batch_size=16, random_state=123).fit(X, y)

    assert np.allclose(m1.weights, m2.weights)
    assert np.isclose(m1.bias, m2.bias)
    assert np.allclose(m1.loss_history, m2.loss_history)


def test_batch_size_larger_than_dataset(binary_data):
    X, y = binary_data
    model = LogisticRegression(lr=0.1, epoch=20, batch_size=10_000, random_state=0).fit(X, y)
    preds = model.predict(X)
    assert preds.shape == (len(X),)


# ---------------------------------------------------------------------------
# Binary Classification Tests
# ---------------------------------------------------------------------------

def test_binary_fit_attributes_and_loss_decrease(binary_data):
    X, y = binary_data
    model = LogisticRegression(lr=0.1, epoch=50, batch_size=16, random_state=42)
    returned = model.fit(X, y)

    assert returned is model
    assert model.weights.shape == (2, 1)
    assert len(model.loss_history) == 50
    assert model.loss_history[-1] < model.loss_history[0]


def test_binary_predict_proba_and_predict_shapes(binary_data):
    X, y = binary_data
    model = LogisticRegression(lr=0.1, epoch=50, batch_size=16, random_state=42).fit(X, y)

    probs = model.predict_proba(X)
    preds = model.predict(X)

    assert probs.shape == (len(X),)
    assert np.all((probs >= 0.0) & (probs <= 1.0))
    assert preds.shape == (len(X),)
    assert set(np.unique(preds)).issubset({0, 1})


def test_binary_classification_accuracy(binary_data):
    X, y = binary_data
    model = LogisticRegression(lr=0.1, epoch=100, batch_size=16, random_state=42).fit(X, y)
    accuracy = np.mean(model.predict(X) == y)

    assert accuracy >= 0.95


def test_binary_threshold_effect(binary_data):
    X, y = binary_data
    m_low = LogisticRegression(lr=0.1, epoch=30, threshold=0.1, random_state=42).fit(X, y)
    m_high = LogisticRegression(lr=0.1, epoch=30, threshold=0.9, random_state=42).fit(X, y)

    assert np.sum(m_low.predict(X)) >= np.sum(m_high.predict(X))


# ---------------------------------------------------------------------------
# Multi-Class Classification Tests
# ---------------------------------------------------------------------------

def test_multiclass_fit_attributes_and_loss_decrease(multiclass_data):
    X, y = multiclass_data
    model = LogisticRegression(lr=0.1, epoch=50, batch_size=16, random_state=42).fit(X, y)

    assert model.weights.shape == (2, 3)
    assert len(model.loss_history) == 50
    assert model.loss_history[-1] < model.loss_history[0]


def test_multiclass_predict_proba_and_predict_shapes(multiclass_data):
    X, y = multiclass_data
    model = LogisticRegression(lr=0.1, epoch=50, batch_size=16, random_state=42).fit(X, y)

    probs = model.predict_proba(X)
    preds = model.predict(X)

    assert probs.shape == (len(X), 3)
    assert np.allclose(np.sum(probs, axis=1), np.ones(len(X)))
    assert preds.shape == (len(X),)
    assert set(np.unique(preds)).issubset({0, 1, 2})


def test_multiclass_classification_accuracy(multiclass_data):
    X, y = multiclass_data
    model = LogisticRegression(lr=0.1, epoch=100, batch_size=16, random_state=42).fit(X, y)
    accuracy = np.mean(model.predict(X) == y)

    assert accuracy >= 0.95