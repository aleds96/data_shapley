from collections import Counter
from typing import Tuple, Union

import numpy as np


class Classifier:
    
    def __init__(self, n_neighbors: int = 3, distance_type: str = 'euclid'):
        if n_neighbors < 1:
            raise ValueError(f"n_neighbors dev esser >= 1, ricevuto {n_neighbors}")
        if distance_type not in ['euclid', 'manhattan']:
            raise ValueError(f"tipo distanza deve ess 'euclid' o 'manhatan', ricevuto {distance_type}")
        
        self.n_neighbors = n_neighbors
        self.distance_type = distance_type
        self.data_x : np.ndarray | None = None
        self.data_y : np.ndarray | None = None
        self._fitted = False
    
    def train(self, X: np.ndarray, y: np.ndarray) -> 'Classifier':
        #allenamento del modello
        X = np.asarray(X, dtype=float)
        y = np.asarray(y)
        
        if X.shape[0] != y.shape[0]:
            raise ValueError(f"Non corrisponde le forme: X {X.shape[0]}, y {y.shape[0]}")
        if X.ndim != 2:
            raise ValueError(f"X deve esser 2D, ricevuto {X.ndim}D")
        if self.n_neighbors > X.shape[0]:
            raise ValueError(f"vicini {self.n_neighbors} > campioni disponibili {X.shape[0]}")
        
        self.data_x : np.ndarray | None = X
        self.data_y : np.ndarray | None = y
        self._fitted = True
        return self
    
    def apply(self, X: np.ndarray) -> np.ndarray:
        #predizione sulla nuvi istanze
        if not self._fitted:
            raise RuntimeError("Devi chiama train() prima di predicere")
        if self.data_x is None or self.data_y is None:
            raise RuntimeError("Dati d'allenamento non sono setati")
        X = np.asarray(X, dtype=float)
        if X.ndim == 1:
            X = X.reshape(1, -1)
        if X.shape[1] != self.data_x.shape[1]:
            raise ValueError(f"Caratteristiche non corrispondono: atteso {self.data_x.shape[1]}, ricevuto {X.shape[1]}")
        
        # calcolo delle previsioni per ogni campione
        results = []
        for sample in X:
            dists = self._calc_dists(sample)
            nearest = self._fetch_nearest(dists)
            labels = self.data_y[nearest]
            pred = self._vote(labels)
            results.append(pred)
        
        return np.array(results)
    
    def get_confidence(self, X: np.ndarray) -> Tuple[np.ndarray, list]:
        #calcolo della confidenza per ogni classe
        if not self._fitted:
            raise RuntimeError("Devi chiama train() per avere confidenza")
        if self.data_x is None or self.data_y is None:
            raise RuntimeError("Dati d'allenamente non trovati")
        X = np.asarray(X, dtype=float)
        if X.ndim == 1:
            X = X.reshape(1, -1)
        if X.shape[1] != self.data_x.shape[1]:
            raise ValueError(f"Numro di caretteristiche sbagliato: atteso {self.data_x.shape[1]}, ricevuto {X.shape[1]}")
        
        #estraggo le classi unice dal traininig
        classes = sorted(np.unique(self.data_y))
        class_map = {c: i for i, c in enumerate(classes)}
        
        confs = []
        for sample in X:
            dists = self._calc_dists(sample)
            nearest = self._fetch_nearest(dists)
            labels = self.data_y[nearest]
            
            counts = Counter(labels)
            conf = np.zeros(len(classes))
            for cls, cnt in counts.items():
                conf[class_map[cls]] = cnt / self.n_neighbors
            confs.append(conf)
        
        return np.array(confs), classes
    
    def _calc_dists(self, sample: np.ndarray) -> np.ndarray:
        #calcolo le distanze rispeto ai vecini nel training set
        if self.distance_type == 'euclid':
            return self._euclid(sample)
        elif self.distance_type == 'manhattan':
            return self._manhat(sample)
        return self._euclid(sample)  
    
    def _euclid(self, sample: np.ndarray) -> np.ndarray:
        #distanza euclidea: sqrt della somma dei quadrati
        diff = self.data_x - sample
        return np.sqrt(np.sum(diff ** 2, axis=1))
    
    def _manhat(self, sample: np.ndarray) -> np.ndarray:
        #distanza manhattan: somma dei valori assoluti
        diff = self.data_x - sample
        return np.sum(np.abs(diff), axis=1)
    
    def _fetch_nearest(self, dists: np.ndarray) -> np.ndarray:
        # ricerca i k vicini piu vicini ordinati per distanza
        return np.argsort(dists)[:self.n_neighbors]
    
    def _vote(self, labels: np.ndarray) -> Union[int, str]:
        #votazione per eticheta piu frequente
        return Counter(labels).most_common(1)[0][0]
    
    def evaluate(self, X: np.ndarray, y: np.ndarray) -> float:
        #calcolo l'accuratezza sul test set
        preds = self.apply(X)
        return np.mean(preds == y)


class Regressor:
    
    def __init__(self, n_neighbors: int = 3, distance_type: str = 'euclid'):
        if n_neighbors < 1:
            raise ValueError(f"n_neighbors dev esser >= 1, {n_neighbors}")
        if distance_type not in ['euclid', 'manhattan']:
            raise ValueError(f"tipo distanza deve ess 'euclid' o 'manhatan', {distance_type}")
        
        self.n_neighbors = n_neighbors
        self.distance_type = distance_type
        self.data_x : np.ndarray | None = None
        self.data_y : np.ndarray | None = None
        self._fitted = False
    
    def train(self, X: np.ndarray, y: np.ndarray) -> 'Regressor':
        #allenamento regressore memorizzazione dati
        X = np.asarray(X, dtype=float)
        y = np.asarray(y, dtype=float)
        
        if X.shape[0] != y.shape[0]:
            raise ValueError(f"Non corrisponde le forme: X {X.shape[0]}, y {y.shape[0]}")
        if X.ndim != 2:
            raise ValueError(f"X deve esser 2D, ricevuto {X.ndim}D")
        if self.n_neighbors > X.shape[0]:
            raise ValueError(f"vicini {self.n_neighbors} > campioni disponibili {X.shape[0]}")
        
        self.data_x : np.ndarray | None = X
        self.data_y : np.ndarray | None = y
        self._fitted = True
        return self
    
    def apply(self, X: np.ndarray) -> np.ndarray:
        #predizione dei valori continui
        if not self._fitted:
            raise RuntimeError("Chiama train() prima di predicere!!")
        if self.data_x is None or self.data_y is None:
            raise RuntimeError("Dati d'allenamento non sono setati")
        X = np.asarray(X, dtype=float)
        if X.ndim == 1:
            X = X.reshape(1, -1)
        if X.shape[1] != self.data_x.shape[1]:
            raise ValueError(f"Numro di caretteristiche sbagliato: atteso {self.data_x.shape[1]}, ricevuto {X.shape[1]}")
        
        #media dei valori dei k vicini piu vicini
        preds = []
        for sample in X:
            diff = self.data_x - sample
            
            if self.distance_type == 'euclid':
                dists = np.sqrt(np.sum(diff ** 2, axis=1))
            elif self.distance_type == 'manhattan':
                dists = np.sum(np.abs(diff), axis=1)
            
            nearest = np.argsort(dists)[:self.n_neighbors]
            pred = np.mean(self.data_y[nearest])
            preds.append(pred)
        
        return np.array(preds)
    
    def evaluate(self, X: np.ndarray, y: np.ndarray) -> float:
        #calcolo score sul test
        preds = self.apply(X)
        ss_res = np.sum((y - preds) ** 2)
        ss_tot = np.sum((y - np.mean(y)) ** 2)
        return 1 - (ss_res / ss_tot)