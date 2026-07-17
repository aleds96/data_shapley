
from typing import List, Tuple, Dict

import numpy as np
import matplotlib.pyplot as plt

from .knn import Classifier

def set_seed(seed: int = 42) -> None:
    np.random.seed(seed)

def _acc_at_fraction(removal: Dict, key: str, frac: float) -> float:
    fr = removal["fraction"]
    if frac in fr:
        return removal[key][fr.index(frac)]
    #frazione più vicina
    i = int(np.argmin(np.abs(np.array(fr) - frac)))
    return removal[key][i]

def accuracy(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.mean(np.asarray(y_true) == np.asarray(y_pred)))

#parte splitting cv
def stratified_split(
    X: np.ndarray, y: np.ndarray, test_size: float = 0.2, seed: int = 42
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    rng = np.random.RandomState(seed)
    X, y = np.asarray(X), np.asarray(y)
    train_idx: List[int] = []
    test_idx: List[int] = []
    for cls in np.unique(y):
        idx = np.where(y == cls)[0]
        rng.shuffle(idx)
        n_test = max(1, int(len(idx) * test_size))
        test_idx.extend(idx[:n_test].tolist())
        train_idx.extend(idx[n_test:].tolist())
    rng.shuffle(train_idx)
    rng.shuffle(test_idx)
    return X[train_idx], y[train_idx], X[test_idx], y[test_idx]

#dividiamo in stratificando in train / validation / test
def split_3way(
    X: np.ndarray, y: np.ndarray,
    train_size: float = 0.6, val_size: float = 0.2, seed: int = 42,
) -> Tuple[np.ndarray, ...]:
    X_train, y_train, X_temp, y_temp = stratified_split(
        X, y, test_size=1 - train_size, seed=seed
    )
    val_fraction = val_size / (1 - train_size)
    X_val, y_val, X_test, y_test = stratified_split(
        X_temp, y_temp, test_size=1 - val_fraction, seed=seed + 1
    )
    return X_train, y_train, X_val, y_val, X_test, y_test


def stratified_kfold(y: np.ndarray, k: int = 5, seed: int = 42) -> List[np.ndarray]:
    rng = np.random.RandomState(seed)
    folds: List[List[int]] = [[] for _ in range(k)]
    for cls in np.unique(y):
        idx = np.where(y == cls)[0]
        rng.shuffle(idx)
        parts = np.array_split(idx, k)
        for i in range(k):
            folds[i].extend(parts[i].tolist())
    return [np.array(sorted(f), dtype=int) for f in folds]

#Seleziona k tramite 5-fold CV stratificata interna al training set."""
def select_best_k(
    X: np.ndarray, y: np.ndarray, candidate_k: List[int],
    seed: int = 7, classifier_cls=Classifier,
) -> Tuple[int, float]:
    
    folds = stratified_kfold(y, k=5, seed=seed)
    best_k, best_score = candidate_k[0], -np.inf
    for k_val in candidate_k:
        scores = []
        for i, val_idx in enumerate(folds):
            train_idx = np.hstack([folds[j] for j in range(len(folds)) if j != i])
            clf = classifier_cls(n_neighbors=k_val).train(X[train_idx], y[train_idx])
            scores.append(accuracy(y[val_idx], clf.apply(X[val_idx])))
        mean_score = float(np.mean(scores))
        if mean_score > best_score:
            best_score, best_k = mean_score, k_val
    return best_k, best_score
#qui controlliamo con noise_frac la quantitò di rumore nelle label che essendo classificaizone si prvede flipping semplice
def inject_label_noise(
    y: np.ndarray, noise_fraction: float = 0.15, seed: int = 99
) -> Tuple[np.ndarray, np.ndarray]:
    #(y_rumorose, indici_flippati).
    rng = np.random.default_rng(seed)
    y_noisy = np.array(y)
    n = len(y_noisy)
    n_flip = int(np.floor(n * noise_fraction))
    if n_flip == 0:
        return y_noisy, np.array([], dtype=int)
    flip_idx = rng.choice(n, size=n_flip, replace=False)
    classes = np.unique(y)
    for idx in flip_idx:
        others = classes[classes != y_noisy[idx]]
        y_noisy[idx] = rng.choice(others)
    return y_noisy, flip_idx

#creazione Dataset sintetico
def make_synthetic_dataset(
    n_samples: int = 200, n_features: int = 6,
    noise_level: float = 0.0, seed: int = 42,
) -> Tuple[np.ndarray, np.ndarray]:
    #Genera dati binari da una funzione lineare del segnale con label noise opzionale.
    rng = np.random.default_rng(seed)
    X = rng.normal(size=(n_samples, n_features))
    signal = X[:, 0] + 0.8 * X[:, 1] + 0.4 * X[:, 2]
    y = (signal > 0).astype(int)
    if noise_level > 0:
        flip_mask = rng.random(n_samples) < noise_level
        y = np.where(flip_mask, 1 - y, y)
    return X, y


def plot_shapley_distribution(values: np.ndarray, title: str, save_path=None) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    axes[0].hist(values, bins=30, color="steelblue", edgecolor="white", alpha=0.85)
    axes[0].axvline(0, color="red", linestyle="--", alpha=0.7, label="Zero")
    axes[0].axvline(np.median(values), color="orange", linestyle="-",
                    alpha=0.8, label=f"Median={np.median(values):.4f}")
    axes[0].set_xlabel("Shapley value")
    axes[0].set_ylabel("Frequency")
    axes[0].set_title(f"{title} — Histogram")
    axes[0].legend()

    sorted_vals = np.sort(values)
    quantiles = np.linspace(0, 1, len(sorted_vals))
    axes[1].plot(quantiles, sorted_vals, color="darkorange", linewidth=2)
    axes[1].axhline(0, color="red", linestyle="--", alpha=0.7)
    axes[1].set_xlabel("Quantile")
    axes[1].set_ylabel("Shapley value")
    axes[1].set_title(f"{title} — Quantile plot")

    plt.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.show()


def plot_removal_curves(results: Dict[str, List[float]], title: str, save_path=None) -> None:
    fractions = results["fraction"]
    plt.figure(figsize=(10, 6))
    plt.axhline(results["acc_baseline"][0], color="gray", linestyle=":",
                label=f'Baseline: {results["acc_baseline"][0]:.3f}', alpha=0.8)
    plt.plot(fractions, results["acc_remove_lowest"], "g-o",
             label="Remove lowest Shapley", linewidth=2, markersize=7)
    plt.plot(fractions, results["acc_remove_highest"], "r-s",
             label="Remove highest Shapley", linewidth=2, markersize=7)
    rand_mean = np.array(results["acc_remove_random_mean"])
    rand_std = np.array(results["acc_remove_random_std"])
    plt.plot(fractions, rand_mean, "b-^", label="Remove random (mean)",
             linewidth=2, markersize=7)
    plt.fill_between(fractions, rand_mean - rand_std, rand_mean + rand_std,
                     color="blue", alpha=0.2, label="Random ± std")
    plt.xlabel("Fraction of training data removed")
    plt.ylabel("Test accuracy")
    plt.title(title)
    plt.legend(loc="best")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.show()
def plot_noise_comparison_curves(
    by_noise: Dict[str, Dict], ds_name: str, save_path=None, levels: List[str] = []
) -> None:
    #levels = list(NOISE_LEVELS.keys())
    fig, axes = plt.subplots(1, len(levels), figsize=(5 * len(levels), 4.5),
                             sharey=True)
    for ax, level in zip(axes, levels):
        rem = by_noise[level]["removal"]
        fr = rem["fraction"]
        ax.axhline(rem["acc_baseline"][0], color="gray", linestyle=":", alpha=0.7)
        ax.plot(fr, rem["acc_remove_lowest"], "g-o", label="lowest", linewidth=2)
        ax.plot(fr, rem["acc_remove_highest"], "r-s", label="highest", linewidth=2)
        ax.plot(fr, rem["acc_remove_random_mean"], "b-^", label="random", linewidth=2)
        ax.set_title(f"{level} noise (k={by_noise[level]['best_k']})")
        ax.set_xlabel("Fraction removed")
        ax.grid(True, alpha=0.3)
    axes[0].set_ylabel("Test accuracy")
    axes[0].legend(loc="best")
    fig.suptitle(f"Removal curves per livello di rumore — {ds_name}", y=1.03)
    plt.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.show()
#Barre: frazione di punti rumorosi finiti nel 20% con Shapley più basso."""
def plot_noise_detection_bar(
    all_results: Dict[str, Dict[str, Dict]], save_path=None, 
    levels :List[str] = []
) -> None:
    datasets = list(all_results.keys())
    #levels = [l for l in NOISE_LEVELS if l != "none"]
    width = 0.8 / len(levels)
    x = np.arange(len(datasets))
    plt.figure(figsize=(11, 6))
    for j, level in enumerate(levels):
        vals = []
        for d in datasets:
            det = all_results[d][level]["detection"]
            vals.append(det["n_noise_in_bottom20"] / det["n_noise"] if det else 0)
        plt.bar(x + j * width, vals, width, label=level)
    plt.xticks(x + width * (len(levels) - 1) / 2, datasets)
    plt.ylabel("Frazione di rumorosi nel 20% più basso")
    plt.ylim(0, 1)
    plt.title("Capacità del Data Shapley di identificare label rumorose")
    plt.legend(title="Noise")
    plt.grid(True, axis="y", alpha=0.3)
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.show()

#Disegna un unico quantile plot con più curve, una per ciascun livello di rumore.
def plot_shapley_quantiles_by_noise(
    all_results: Dict[str, Dict[str, Dict]],
    ds_name: str,
    levels: List[str],
    save_path: str = None,
) -> None:
    plt.figure(figsize=(8,6))
    
    for level in levels:
        vals = all_results[ds_name][level]["shapley_values"]
        sorted_vals = np.sort(vals)
        quantiles = np.linspace(0, 1, len(sorted_vals))
        plt.plot(quantiles, sorted_vals, linewidth=2, label=f"{level} noise")
    
    plt.axhline(0, color="red", linestyle="--", alpha=0.7)
    plt.xlabel("Quantile")
    plt.ylabel("Shapley value")
    plt.title(f"Quantile plot Shapley — {ds_name}")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.show()
