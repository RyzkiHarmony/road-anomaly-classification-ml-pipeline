# normalizer.py
# Modul normalisasi Z-score global yang diikat pada statistik training set.
# Mencegah data leakage antar-fold dan memastikan inferensi edge konsisten.

import json
import numpy as np


class GlobalZScoreNormalizer:
    """Normalisasi Global Z-Score untuk sinyal multivariat 3D (N, C, T) atau 2D.
    
    Menghitung mean dan std per kanal dari training data, kemudian
    mentransformasikan validation/test data dengan parameter yang sama.
    """
    def __init__(self, eps: float = 1e-6):
        self.eps = eps
        self.means = None
        self.stds = None

    def fit(self, X: np.ndarray):
        """Fit scaler pada data latih (N, C, T) atau (N, F)."""
        if X.ndim == 3:
            # Over samples (axis 0) and time (axis 2) -> shape (1, C, 1)
            self.means = np.mean(X, axis=(0, 2), keepdims=True)
            self.stds = np.std(X, axis=(0, 2), keepdims=True)
        elif X.ndim == 2:
            self.means = np.mean(X, axis=0, keepdims=True)
            self.stds = np.std(X, axis=0, keepdims=True)
        else:
            raise ValueError(f"Dimensi array tidak didukung: {X.ndim}")
        
        self.stds = np.where(self.stds < self.eps, 1.0, self.stds)
        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        """Transform data menggunakan mean dan std dari training set."""
        if self.means is None or self.stds is None:
            raise RuntimeError("Normalizer belum di-fit!")
        return (X - self.means) / self.stds

    def fit_transform(self, X: np.ndarray) -> np.ndarray:
        return self.fit(X).transform(X)

    def save_json(self, filepath: str):
        """Simpan parameter scaler ke format JSON untuk mobile edge deployment."""
        data = {
            "means": self.means.flatten().tolist(),
            "stds": self.stds.flatten().tolist(),
            "eps": self.eps
        }
        with open(filepath, "w") as f:
            json.dump(data, f, indent=4)

    def load_json(self, filepath: str, shape=(1, -1, 1)):
        """Muat parameter scaler dari file JSON."""
        with open(filepath, "r") as f:
            data = json.load(f)
        self.means = np.array(data["means"]).reshape(shape)
        self.stds = np.array(data["stds"]).reshape(shape)
        self.eps = data.get("eps", 1e-6)
        return self
