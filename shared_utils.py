"""Shared utilities for Part 2 and Part 4.

Contains stratified splitting, feature standardisation, logistic regression,
K-nearest neighbours, and classification metrics.
"""
import csv
import math
import numpy as np

FEATURE_NAMES = [
    "customer_tenure_months",
    "complaint_count_90d",
    "discount_order_share",
    "avg_order_value_inr_hundreds",
    "loyalty_score",
    "cod_flag",
    "plus_member_flag"
]


def load_dataset(csv_path="data/records.csv"):
    with open(csv_path, "r", newline="") as f:
        reader = csv.DictReader(f)
        rows = list(reader)
    return rows


def make_stratified_split(rows):
    """Make stratified 70/30 split keeping file order.
    Rows with outcome 0: first floor(7 * count / 10) to train, rest to test.
    Rows with outcome 1: first floor(7 * count / 10) to train, rest to test.
    Both sets keep original file order.
    """
    rows_0 = [(idx, r) for idx, r in enumerate(rows) if int(r["churned"]) == 0]
    rows_1 = [(idx, r) for idx, r in enumerate(rows) if int(r["churned"]) == 1]

    n0_train = (7 * len(rows_0)) // 10
    n1_train = (7 * len(rows_1)) // 10

    train_pairs = sorted(rows_0[:n0_train] + rows_1[:n1_train], key=lambda x: x[0])
    test_pairs = sorted(rows_0[n0_train:] + rows_1[n1_train:], key=lambda x: x[0])

    train_rows = [p[1] for p in train_pairs]
    test_rows = [p[1] for p in test_pairs]

    return train_rows, test_rows


def extract_features_and_targets(train_rows, test_rows):
    """Extract raw X1-X7 and targets, standardise using training mean and sample SD."""
    x_train_raw = np.array([[float(r[col]) for col in FEATURE_NAMES] for r in train_rows])
    y_train = np.array([float(r["churned"]) for r in train_rows])

    x_test_raw = np.array([[float(r[col]) for col in FEATURE_NAMES] for r in test_rows])
    y_test = np.array([float(r["churned"]) for r in test_rows])

    train_mean = np.mean(x_train_raw, axis=0)
    train_sd = np.std(x_train_raw, axis=0, ddof=1)

    x_train_std = (x_train_raw - train_mean) / train_sd
    x_test_std = (x_test_raw - train_mean) / train_sd

    return x_train_raw, y_train, x_test_raw, y_test, x_train_std, x_test_std, train_mean, train_sd


def sigmoid(z):
    return 1.0 / (1.0 + np.exp(-z))


def bce_loss(y_true, y_hat):
    # Clamping for safety against log(0) in general expressions, though sigmoid values are well-behaved here
    eps = 1e-15
    y_clipped = np.clip(y_hat, eps, 1.0 - eps)
    return -float(np.mean(y_true * np.log(y_clipped) + (1.0 - y_true) * np.log(1.0 - y_clipped)))


def train_logistic_regression(x_train, y_train, eta=0.1, epochs=1000):
    """Train logistic regression starting from w=0, b=0 with full-batch gradient descent."""
    n = len(y_train)
    p = x_train.shape[1]
    w = np.zeros(p)
    b = 0.0

    loss_history = {}
    for ep in range(epochs + 1):
        z = x_train @ w + b
        y_hat = sigmoid(z)
        loss = bce_loss(y_train, y_hat)
        loss_history[ep] = loss

        if ep == epochs:
            break

        grad_w = (1.0 / n) * (x_train.T @ (y_hat - y_train))
        grad_b = (1.0 / n) * float(np.sum(y_hat - y_train))

        w -= eta * grad_w
        b -= eta * grad_b

    return w, b, loss_history


def compute_confusion_and_metrics(y_true, probas, tau):
    """Compute confusion matrix and classification metrics at threshold tau."""
    preds = (probas >= tau).astype(int)
    tp = int(np.sum((preds == 1) & (y_true == 1)))
    fp = int(np.sum((preds == 1) & (y_true == 0)))
    tn = int(np.sum((preds == 0) & (y_true == 0)))
    fn = int(np.sum((preds == 0) & (y_true == 1)))

    n = len(y_true)
    acc = (tp + tn) / n if n > 0 else 0.0
    prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    rec = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    spec = tn / (tn + fp) if (tn + fp) > 0 else 0.0
    f1 = (2.0 * prec * rec / (prec + rec)) if (prec + rec) > 0 else 0.0

    return {
        "tp": tp,
        "fp": fp,
        "tn": tn,
        "fn": fn,
        "accuracy": acc,
        "precision": prec,
        "recall": rec,
        "f1": f1,
        "specificity": spec
    }


def knn_predict_query(query, x_ref, y_ref, k):
    """Predict for a single query vector using Euclidean distance and stable sort.
    Returns predicted class (1 or 0) and probability (fraction of positive neighbours).
    """
    dists = np.sqrt(np.sum((x_ref - query) ** 2, axis=1))
    order = np.argsort(dists, kind="stable")
    top_k_labels = y_ref[order[:k]]
    pos_count = np.sum(top_k_labels)
    proba = pos_count / k
    pred = 1 if pos_count >= (k + 1) / 2 else 0
    return pred, proba


def knn_predict(x_query, x_ref, y_ref, k):
    """Predict for multiple query records."""
    preds = []
    probas = []
    for q in x_query:
        pred, prob = knn_predict_query(q, x_ref, y_ref, k)
        preds.append(pred)
        probas.append(prob)
    return np.array(preds), np.array(probas)
