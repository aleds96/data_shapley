from typing import List, Optional
import numpy as np
import warnings
from .knn import Classifier
from .utils import accuracy

def _score_subset(
    X_subset: np.ndarray,
    y_subset: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    n_neighbors: int,
) -> float:
    
    #Allena un classificatore su X_subset/y_subset e restituisce l'accuracy su X_val.
    if len(X_subset) == 0:
        return 0.0
    
    k_eff = min(n_neighbors, max(1, len(X_subset)))
    
    #if k_eff < n_neighbors:
    #f"_score_subset: uso k={n_neighbors} come vicini > dispo.={len(X_subset)}, usando quindi==> k_eff={k_eff}"
    
    clf = Classifier(n_neighbors=k_eff)
    clf.train(X_subset, y_subset)
    preds = clf.apply(X_val)
    return float(accuracy(y_val, preds))

#Stima Shapley
#Restituisce array di lunghezza len(X_train) con score per ogni punto di training
def estimate_shapley_values(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    n_permutations: int = 200,
    n_neighbors: int = 5,
    random_state: int = 42
) -> np.ndarray:
   
    rng = np.random.default_rng(random_state)
    n_samples = len(X_train)
    values = np.zeros(n_samples, dtype=float)

    for _ in range(max(1, int(n_permutations))):
        ordering = rng.permutation(n_samples).astype(int) 
        subset: List[int] = []
        for idx in ordering:
            subset.append(int(idx))            
            prev_score = _score_subset(
            X_train[np.array(subset[:-1], dtype=int)],
            y_train[np.array(subset[:-1], dtype=int)],
            X_val,
            y_val,
            n_neighbors
            )
            new_score = _score_subset(
                X_train[np.array(subset, dtype=int)],
                y_train[np.array(subset, dtype=int)],
                X_val,
                y_val,
                n_neighbors
            )

            values[idx] += new_score - prev_score

    values /= max(1, int(n_permutations))
    return values
