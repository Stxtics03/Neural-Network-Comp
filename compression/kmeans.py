"""
Standard Lloyd's-algorithm k-means, with a choice of initialization:

- "random": pick k initial centroids uniformly at random from the data
  points. The classic weak point of k-means -- a bad random draw can
  converge to a poor local optimum, and results vary a lot run to run.
- "kmeans++": pick initial centroids one at a time, each drawn with
  probability proportional to its squared distance from the nearest
  already-chosen centroid. This spreads initial centroids out and is
  the standard "smart init" used by scikit-learn and most production
  k-means implementations -- it already solves much of the bad-init
  problem DA is meant to solve, which is why comparing DA against
  random-init k-means (not just kmeans++) matters: kmeans++ is the
  harder baseline to beat, precisely because it isn't a naive one.
"""
from __future__ import annotations

import numpy as np


def _kmeans_plusplus_init(X: np.ndarray, k: int, rng: np.random.Generator) -> np.ndarray:
    n = X.shape[0]
    centroids = [X[rng.integers(n)]]
    for _ in range(1, k):
        dists = np.min(
            [((X - c) ** 2).sum(axis=1) for c in centroids], axis=0
        )
        probs = dists / (dists.sum() + 1e-12)
        next_idx = rng.choice(n, p=probs)
        centroids.append(X[next_idx])
    return np.stack(centroids)


def _random_init(X: np.ndarray, k: int, rng: np.random.Generator) -> np.ndarray:
    idx = rng.choice(X.shape[0], size=k, replace=False)
    return X[idx].copy()


class KMeansClusterer:
    def __init__(
        self, k: int, init: str = "kmeans++", max_iters: int = 300,
        tol: float = 1e-5, seed: int = 0, n_init: int = 1,
    ):
        if init not in ("random", "kmeans++"):
            raise ValueError(f"init must be 'random' or 'kmeans++', got {init}")
        self.k = k
        self.init = init
        self.max_iters = max_iters
        self.tol = tol
        self.n_init = n_init  # number of independent restarts; keep the best by inertia
        self.rng = np.random.default_rng(seed)
        self.centroids_: np.ndarray | None = None
        self.labels_: np.ndarray | None = None
        self.total_iters_: int = 0  # summed across all restarts, for compute-parity checks

    def _fit_once(self, X: np.ndarray, k_target: int) -> tuple[np.ndarray, np.ndarray, float, int]:
        centroids = (
            _kmeans_plusplus_init(X, k_target, self.rng) if self.init == "kmeans++"
            else _random_init(X, k_target, self.rng)
        )
        n_iters = 0
        for _ in range(self.max_iters):
            dists = ((X[:, None, :] - centroids[None, :, :]) ** 2).sum(axis=2)
            labels = dists.argmin(axis=1)

            new_centroids = centroids.copy()
            for c in range(k_target):
                members = X[labels == c]
                if len(members) > 0:
                    new_centroids[c] = members.mean(axis=0)
                # empty cluster: leave centroid where it was rather than
                # reseeding -- simpler, and rare in practice for this
                # data size/k range.

            shift = np.linalg.norm(new_centroids - centroids)
            centroids = new_centroids
            n_iters += 1
            if shift < self.tol:
                break

        dists = ((X[:, None, :] - centroids[None, :, :]) ** 2).sum(axis=2)
        labels = dists.argmin(axis=1)
        inertia = dists[np.arange(len(X)), labels].sum()  # within-cluster SSE; lower is better
        return centroids, labels, inertia, n_iters

    def fit(self, X: np.ndarray) -> "KMeansClusterer":
        n = X.shape[0]
        k_target = min(self.k, n)  # can't have more clusters than points

        best_centroids, best_labels, best_inertia = None, None, np.inf
        self.total_iters_ = 0
        for _ in range(self.n_init):
            centroids, labels, inertia, n_iters = self._fit_once(X, k_target)
            self.total_iters_ += n_iters
            if inertia < best_inertia:
                best_centroids, best_labels, best_inertia = centroids, labels, inertia

        self.centroids_ = best_centroids
        self.labels_ = best_labels
        return self
