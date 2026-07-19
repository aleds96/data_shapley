from typing import Dict, List
import numpy as np

from .knn import Classifier
from .utils import accuracy, split_3way, select_best_k, inject_label_noise
from .shapley_score import estimate_shapley_values

#valuta come cambia accuracy sul test rimuovendo frazione (lowest/hightest/causalee )
def evaluate_removal_experiment(
    X_train: np.ndarray, y_train: np.ndarray,
    X_test: np.ndarray, y_test: np.ndarray,
    shapley_values: np.ndarray, n_neighbors: int,
    fractions: List[float], n_random_repeats: int = 20, seed: int = 42,
) -> Dict[str, List[float]]:
    rng = np.random.RandomState(seed)
    n_train = len(X_train)
    order_low = np.argsort(shapley_values)
    order_high = order_low[::-1]

    results: Dict[str, List[float]] = {
        "fraction": [], "acc_baseline": [],
        "acc_remove_lowest": [], "acc_remove_highest": [],
        "acc_remove_random_mean": [], "acc_remove_random_std": [],
    }

    baseline = accuracy(
        y_test, Classifier(n_neighbors).train(X_train, y_train).apply(X_test)
    )

    def _fit_eval(keep_mask: np.ndarray) -> float:
        k_eff = min(n_neighbors, int(keep_mask.sum()))
        clf = Classifier(k_eff).train(X_train[keep_mask], y_train[keep_mask])
        return accuracy(y_test, clf.apply(X_test))

    for frac in fractions:
        n_remove = int(np.floor(n_train * frac))
        results["fraction"].append(frac)
        results["acc_baseline"].append(baseline)

        if n_remove == 0:
            results["acc_remove_lowest"].append(baseline)
            results["acc_remove_highest"].append(baseline)
            results["acc_remove_random_mean"].append(baseline)
            results["acc_remove_random_std"].append(0.0)
            continue

        mask = np.ones(n_train, bool); 
        mask[order_low[:n_remove]] = False
        results["acc_remove_lowest"].append(_fit_eval(mask))

        mask = np.ones(n_train, bool); 
        mask[order_high[:n_remove]] = False
        results["acc_remove_highest"].append(_fit_eval(mask))

        rand_accs = []
        for _ in range(n_random_repeats):
            mask = np.ones(n_train, bool)
            mask[rng.choice(n_train, size=n_remove, replace=False)] = False
            rand_accs.append(_fit_eval(mask))
        results["acc_remove_random_mean"].append(float(np.mean(rand_accs)))
        results["acc_remove_random_std"].append(float(np.std(rand_accs)))

    return results


#qui pipeline (dataset,rumore)
def run_single(
    X: np.ndarray, y: np.ndarray, *,
    noise_fraction: float, candidate_k: List[int], fractions: List[float],
    n_permutations: int, n_random_repeats: int, seed: int = 42,
) -> Dict:
    
    #Esegue split, aggiunta rumore (solo sul training), set k,
    #individuazionee Shapley. Ritorna un dizionario di risultati.
    
    X_tr, y_tr, X_val, y_val, X_te, y_te = split_3way(
        X, y, train_size=0.6, val_size=0.2, seed=seed
    )

    if noise_fraction > 0:
        y_tr_used, noise_idx = inject_label_noise(y_tr, noise_fraction, seed=99)
    else:
        y_tr_used, noise_idx = y_tr.copy(), np.array([], dtype=int)

    best_k, cv_score = select_best_k(X_tr, y_tr_used, candidate_k, seed=7)
    #print('best k==>',type(best_k))
    shapley = estimate_shapley_values(
        X_tr, y_tr_used, X_val, y_val,
        n_permutations=n_permutations, n_neighbors=best_k, random_state=seed,
    )

    removal = evaluate_removal_experiment(
        X_tr, y_tr_used, X_te, y_te, shapley, best_k,
        fractions, n_random_repeats, seed=seed,
    )

    #Metriche di detection del rumore
    detection = None
    if len(noise_idx) > 0:
        ranks = np.empty_like(np.argsort(shapley))
        ranks[np.argsort(shapley)] = np.arange(len(shapley))
        #qui mettiamo 20 per la regola 80-20 ma in realtà potremmo decidere in base ai quantili
        thr = int(0.2 * len(shapley))
        detection = {
            "n_noise": int(len(noise_idx)),
            "median_rank_noise": float(np.median(ranks[noise_idx])),
            "n_noise_in_bottom20": int(np.sum(ranks[noise_idx] < thr)),
        }

    return {
        "best_k": int(best_k),
        "cv_score": float(cv_score),
        "shapley_values": shapley,
        "removal": removal,
        "noise_idx": noise_idx,
        "detection": detection,
        "noise_fraction": noise_fraction,
    }




def run_all_noise_levels(
    X: np.ndarray, y: np.ndarray,
    candidate_k: List[int], fractions: List[float],
    n_permutations: int, n_random_repeats: int, seed: int = 42,
    noise_levels: Dict[str, float] = {}
) -> Dict[str, Dict]:
    return {
        level: run_single(
            X, y, noise_fraction=frac, candidate_k=candidate_k,
            fractions=fractions, n_permutations=n_permutations,
            n_random_repeats=n_random_repeats, seed=seed,
        )
        for level, frac in noise_levels.items()
    }
