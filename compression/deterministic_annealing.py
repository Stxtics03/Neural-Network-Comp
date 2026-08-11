"""
Deterministic annealing (DA) clustering (Rose, 1998).

The idea: instead of k-means' hard assignment (each point belongs to
exactly one cluster, decided by nearest centroid), DA uses a SOFT
assignment controlled by a temperature T:

    p(cluster c | point x) = exp(-dist(x, c)^2 / T) / sum_c' exp(-dist(x, c')^2 / T)

At high T, this is close to uniform -- every point contributes a
little to every centroid, so the initial centroid positions barely
matter (this is the mechanism that's supposed to make DA more robust
to bad initialization than k-means, whose hard assignment can lock
into a bad local optimum from a bad starting point). As T is cooled
toward 0, the soft assignment sharpens toward a hard one-hot
assignment, and DA's behavior converges to k-means' behavior.

Algorithm: start at a high temperature, run soft-assignment + weighted
centroid update to convergence, cool T by a fixed rate, repeat until T
reaches a minimum, then take a final hard assignment.
"""
from __future__ import annotations

import math

import numpy as np

from compression.kmeans import _kmeans_plusplus_init


class DeterministicAnnealingClusterer:
    def __init__(
        self, k: int, t_init: float = 10.0, t_min: float = 1e-3,
        cooling_rate: float = 0.9, max_iters_per_temp: int = 50,
        tol: float = 1e-5, seed: int = 0,
    ):
        self.k = k
        self.t_init = t_init
        self.t_min = t_min
        self.cooling_rate = cooling_rate
        self.max_iters_per_temp = max_iters_per_temp
        self.tol = tol
        self.rng = np.random.default_rng(seed)
        self.centroids_: np.ndarray | None = None
        self.labels_: np.ndarray | None = None
        self.total_iters_: int = 0

    def _soft_assign_probs(self, X: np.ndarray, centroids: np.ndarray, T: float) -> np.ndarray:
        sq_dists = ((X[:, None, :] - centroids[None, :, :]) ** 2).sum(axis=2)
        # Subtract the row-min before exponentiating -- pure numerical
        # stability, doesn't change the resulting probabilities (softmax
        # is shift-invariant), just avoids exp() overflow for large
        # negative exponents.
        neg_scaled = -(sq_dists - sq_dists.min(axis=1, keepdims=True)) / T
        exp_vals = np.exp(neg_scaled)
        return exp_vals / exp_vals.sum(axis=1, keepdims=True)

    def _converge_at_temperature(self, X: np.ndarray, centroids: np.ndarray, T: float) -> np.ndarray:
        for _ in range(self.max_iters_per_temp):
            probs = self._soft_assign_probs(X, centroids, T)  # (n, k)
            weights = probs.sum(axis=0, keepdims=True).T  # (k, 1)
            new_centroids = (probs.T @ X) / np.maximum(weights, 1e-12)

            shift = np.linalg.norm(new_centroids - centroids)
            centroids = new_centroids
            self.total_iters_ += 1
            if shift < self.tol:
                break
        return centroids

    def _break_symmetry(self, centroids: np.ndarray, data_scale: float) -> np.ndarray:
        """
        At high temperature, DA's soft assignment is theoretically
        supposed to collapse multiple centroids onto the same point
        (the "single effective cluster" phase) -- but once centroids
        are numerically identical, they receive IDENTICAL gradient
        updates forever and can never split apart again as T cools.
        Real DA handles this by explicitly perturbing near-duplicate
        centroids apart (Rose 1998 calls this "codeword splitting").
        Without this step, DA silently returns fewer effective clusters
        than requested -- confirmed via a synthetic 3-blob test where
        omitting this collapsed 3 centroids down to 2.
        """
        centroids = centroids.copy()
        eps = 1e-4 * math.sqrt(data_scale)
        for i in range(len(centroids)):
            for j in range(i + 1, len(centroids)):
                if np.linalg.norm(centroids[i] - centroids[j]) < eps:
                    centroids[j] = centroids[j] + self.rng.normal(
                        scale=eps * 10, size=centroids[j].shape
                    )
        return centroids

    def fit(self, X: np.ndarray) -> "DeterministicAnnealingClusterer":
        n, dim = X.shape
        k_target = min(self.k, n)
        centroids = _kmeans_plusplus_init(X, k_target, self.rng)

        # Temperature is scaled to the data's own variance so t_init/t_min
        # act as relative multipliers of the data's natural distance
        # scale, not absolute values -- weight vectors and, say, pixel
        # intensities live on very different numeric scales, and a fixed
        # absolute T tuned for one would be wildly wrong for the other.
        data_scale = max(((X - X.mean(axis=0)) ** 2).sum(axis=1).mean(), 1e-12)
        t_init_eff = self.t_init * data_scale
        t_min_eff = self.t_min * data_scale

        T = t_init_eff
        while True:
            centroids = self._converge_at_temperature(X, centroids, T)
            centroids = self._break_symmetry(centroids, data_scale)
            if T <= t_min_eff:
                break
            T = max(T * self.cooling_rate, t_min_eff)

        self.centroids_ = centroids
        dists = ((X[:, None, :] - centroids[None, :, :]) ** 2).sum(axis=2)
        self.labels_ = dists.argmin(axis=1)
        return self
