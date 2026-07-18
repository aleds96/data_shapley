
from typing import List, Tuple, Dict,Optional

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

#creazione Dataset sintetico. i primi informative_dim feature contengono il segnale (combinazione lineare), mentre le restanti feature sono o rumore puro o ridondanti.
#"noisy" (aggiunge feature casuali) o "redundant" (duplica informative con rumore)
def make_synthetic_dataset(
    n_samples: int = 200,
    n_features: int = 6,
    noise_level: float = 0.0,
    seed: int = 42,
    informative_dim: int = 3,
    mode: str = "redundant",  # 
) -> Tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    if informative_dim > n_features:
        informative_dim = n_features

    #genera le feature informative
    X_inf = rng.normal(loc=0.0, scale=1.0, size=(n_samples, informative_dim))

    #costruisci il segnale come combinazione lineare delle informative
    #con pesi decrescenti
    weights = np.array([1.0, 0.8, 0.4] + [0.2] * max(0, informative_dim - 3))[:informative_dim]
    signal = X_inf.dot(weights[:informative_dim])

    #etichette binarie dal segnale (soglia 0)
    y = (signal > 0).astype(int)

    #aggiungi feature aggiuntive
    n_extra = n_features - informative_dim
    if n_extra > 0:
        if mode == "redundant":
            #duplicati rumorosi delle informative (mantiene SNR)
            extras = []
            for i in range(n_extra):
                src_col = i % informative_dim
                #aggiungi piccolo rumore gaussiano
                extras.append(X_inf[:, src_col] + 0.1 * rng.normal(size=n_samples))
            X_extra = np.column_stack(extras)
        else:
            #mode == "noisy": (degradano SNR)
            X_extra = rng.normal(size=(n_samples, n_extra))
        X = np.hstack([X_inf, X_extra])
    else:
        X = X_inf
    if noise_level > 0:
        flip_mask = rng.random(n_samples) < float(noise_level)
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
    fig.suptitle(f"Removal curves by noise level — {ds_name}", y=1.03)
    plt.tight_layout()
    if save_path:
        OUT_PATH = save_path / f'noise_comparison_curves_{ds_name}.png'
        plt.savefig(OUT_PATH, dpi=150, bbox_inches="tight")
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
        OUT_PATH = save_path / f'shapley_quantiles_by_noise_{ds_name}.png'
        plt.savefig(OUT_PATH, dpi=150, bbox_inches="tight")
    plt.show()
def compute_quantile_curve(values: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    vals = np.asarray(values)
    if vals.size == 0:
        return np.array([]), np.array([])
    s = np.sort(vals)
    q = np.linspace(0.0, 1.0, len(s))
    return q, s

def plot_quantile_shapley_clean_vs_noisy_by_noise(
    all_results: Dict[str, Dict[str, Dict]],
    ds_name: str,
    levels: List[str],
    *,
    figsize_per_panel: Tuple[int,int] = (4,4),
    colors: Optional[Dict[str,str]] = None,
    save_path: Optional[str] = None,
) -> None:
    if colors is None:
        colors = {"clean":"tab:blue", "noisy":"tab:orange"}

    n = len(levels)
    fig_w = figsize_per_panel[0] * n
    fig_h = figsize_per_panel[1]
    fig, axes = plt.subplots(1, n, figsize=(fig_w, fig_h), squeeze=False)
    axes = axes[0]

    for ax, level in zip(axes, levels):
        entry = all_results[ds_name].get(level, {})
        shap = np.asarray(entry.get("shapley_values", np.array([])))
        shap = normalize_minmax(shap, feature_range=(-1.0, 1.0))
        noise_idx = np.asarray(entry.get("noise_idx", np.array([], dtype=int)))
        n_total = len(shap)

        noisy_mask = np.zeros(n_total, dtype=bool)
        if noise_idx.size > 0:
            valid = noise_idx[(noise_idx >= 0) & (noise_idx < n_total)]
            noisy_mask[valid] = True
        clean_mask = ~noisy_mask

        clean_vals = shap[clean_mask]
        noisy_vals = shap[noisy_mask] if noisy_mask.any() else np.array([])

        q_c, s_c = compute_quantile_curve(clean_vals)
        q_n, s_n = compute_quantile_curve(noisy_vals)

        if s_c.size > 0:
            ax.plot(q_c, s_c, color=colors["clean"], lw=2, label=f"clean (n={len(s_c)})")
        if s_n.size > 0:
            ax.plot(q_n, s_n, color=colors["noisy"], lw=2, linestyle="--", label=f"noisy (n={len(s_n)})")

        ax.axhline(0.0, color="red", linestyle="--", alpha=0.6)
        ax.set_title(level)
        ax.set_xlabel("Quantile")
        ax.set_xlim(0,1)
        ax.grid(True, alpha=0.25)
        ax.legend(fontsize=9)

    fig.suptitle(f"Shapley quantiles: clean vs noisy — {ds_name}", y=1.02)
    plt.tight_layout()
    if save_path:
        OUT_PATH = save_path / f'quantile_clean_vs_noisy_{ds_name}.png'
        plt.savefig(OUT_PATH, dpi=150, bbox_inches="tight")
def normalize_minmax(values,feature_range: Tuple[float,float] = (0.0, 1.0)) -> np.ndarray:
    vals = np.asarray(values, dtype=float)
    if vals.size == 0:
        return vals.copy()
    vmin = np.nanmin(vals)
    vmax = np.nanmax(vals)
    lo, hi = feature_range
    denom = vmax - vmin
    if denom == 0 or np.isnan(denom):
        return np.full_like(vals, fill_value=lo, dtype=float)
    scaled = (vals - vmin) / denom
    return lo + scaled * (hi - lo)