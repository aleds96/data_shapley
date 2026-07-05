import sys
import numpy as np
from knn import Classifier


def test_classification():
    np.random.seed(42)
    
    n_samples = 150
    X = np.random.randn(n_samples, 4) * 2
    y = np.random.choice([0, 1, 2], n_samples)
    
    split = int(0.7 * n_samples)
    X_train, X_test = X[:split], X[split:]
    y_train, y_test = y[:split], y[split:]
    
    clf = Classifier(n_neighbors=5, distance_type='euclid')
    clf.train(X_train, y_train)
    
    preds = clf.apply(X_test)
    acc = clf.evaluate(X_test, y_test)
    
    confs, classes = clf.get_confidence(X_test[:3])
    
    print("Classification Results")
    print("-" * 10)
    print(f"Accuracy: {acc:.4f}")
    print(f"Predictions shape: {preds.shape}")
    print(f"First 5 predictions: {preds[:5]}")
    print(f"Confidence classes: {classes}")
    print(f"Sample confidences:\n{confs[0]}")
    print()


def test_manhattan_distance():
    np.random.seed(42)
    
    X_train = np.array([[0, 0], [1, 1], [2, 2], [5, 5], [6, 6]])
    y_train = np.array([0, 0, 0, 1, 1])
    X_test = np.array([[0.5, 0.5], [5.5, 5.5]])
    
    clf = Classifier(n_neighbors=3, distance_type='manhattan')
    clf.train(X_train, y_train)
    preds = clf.apply(X_test)
    
    print("Manhattan Distance Test")
    print("-" * 40)
    print(f"Train data:\nX: {X_train}\ny: {y_train}")
    print(f"Test predictions: {preds}")
    print()


if __name__ == "__main__":
    test_classification()
    test_manhattan_distance()
    print("All tests ok ")
