from collections import Counter
from typing import Tuple, Union

import numpy as np


class Classifier:

    def __init__(self, n_neighbors: int = 3, distance_type: str = "euclid"):
        if n_neighbors < 1:
            raise ValueError(f"n_neighbors dev esser >= 1, ricevuto {n_neighbors}")
        if distance_type not in ["euclid", "manhattan"]:
            raise ValueError(f"tipo distanza deve ess 'euclid' o 'manhatan', ricevuto {distance_type}")

        self.n_neighbors = n_neighbors
        self.distance_type = distance_type
        self.data_x: np.ndarray | None = None
        self.data_y: np.ndarray | None = None
        self._fitted = False

    def train(self, X: np.ndarray, y: np.ndarray) -> "Classifier":
        X = np.asarray(X, dtype=float)
        y = np.asarray(y)

        if X.shape[0] != y.shape[0]:
            raise ValueError(f"Non corrisponde le forme: X {X.shape[0]}, y {y.shape[0]}")
        if X.ndim != 2:
            raise ValueError(f"X deve esser 2D, ricevuto {X.ndim}D")
        if self.n_neighbors > X.shape[0]:
            raise ValueError(f"vicini {self.n_neighbors} > campioni disponibili {X.shape[0]}")

        self.data_x = X
        self.data_y = y
        self._fitted = True
        return self

    def apply(self, X: np.ndarray) -> np.ndarray:
        self._check_fitted()
        X = np.asarray(X, dtype=float)
        if X.ndim == 1:
            X = X.reshape(1, -1)
        self._check_feature_dim(X)

        dists = self._pairwise_distances(X)
        nearest = np.argpartition(dists, kth=self.n_neighbors - 1, axis=1)[:, : self.n_neighbors]
        nearest = nearest[np.arange(len(X))[:, None], np.argsort(dists[np.arange(len(X))[:, None], nearest], axis=1)]

        labels = self.data_y[nearest]
        return np.array([self._vote(label_row) for label_row in labels])


    def _pairwise_distances(self, X: np.ndarray) -> np.ndarray:
        if self.distance_type == "euclid":
            diff = X[:, None, :] - self.data_x[None, :, :]
            return np.sqrt(np.sum(diff * diff, axis=2))
        diff = np.abs(X[:, None, :] - self.data_x[None, :, :])
        return np.sum(diff, axis=2)
    
    def _check_fitted(self) -> None:
        if not self._fitted or self.data_x is None or self.data_y is None:
            raise RuntimeError("Devi chiama train() prima di predicere")

    def _check_feature_dim(self, X: np.ndarray) -> None:
        if X.shape[1] != self.data_x.shape[1]:
            raise ValueError(f"Caratteristiche non corrispondono: atteso {self.data_x.shape[1]}, ricevuto {X.shape[1]}")

    def _vote(self, labels: np.ndarray) -> Union[int, str]:
        return Counter(labels).most_common(1)[0][0]

