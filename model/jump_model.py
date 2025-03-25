# This file is adapted from Nystrup et al. (2021).
# Source: https://www.sciencedirect.com/science/article/pii/S0957417421009647#appSB (accessed on March 24, 2025)

import numpy as np
from scipy.spatial.distance import cdist

class JumpModel:
    def __init__(self, n_states, jump_penalty=1e-5, max_iter=10, n_init=10, tol=None, verbose=False):
        self.n_states = n_states
        self.jump_penalty = jump_penalty
        self.max_iter = max_iter
        self.n_init = n_init
        self.tol = tol
        self.verbose = verbose
        self.mu = None
        self.Gamma = jump_penalty * (1 - np.eye(n_states))

    def fit(self, Y, initial_states=None):
        self.Y = Y
        self.n_obs, self.n_features = Y.shape

        if initial_states is not None:
            initial_states = np.array(initial_states, dtype=np.int64)
            if len(np.unique(initial_states)) == self.n_states:
                self.s = initial_states.copy()
            else:
                self.s = self._init_states()
        else:
            self.s = self._init_states()

        self.best_loss = None
        self.best_s = None

        for init in range(self.n_init):
            mu = np.zeros((self.n_states, self.n_features))
            loss_old = 1e10

            for it in range(self.max_iter):
                self._m_step(mu)
                loss, converged = self._e_step(mu, loss_old)

                if self.verbose:
                    print(f"Iteration {it}: {loss:.6e}")

                if converged:
                    break

                loss_old = loss

            if self.best_loss is None or loss_old < self.best_loss:
                self.best_loss = loss_old
                self.best_s = self.s.copy()
                self.mu = mu.copy()

            self.s = self._init_states()

        self.s = self.best_s

    def inference(self, new_Y):
        if self.mu is None:
            raise ValueError("Model parameters are not fitted. Call fit() before inference.")

        n_obs, _ = new_Y.shape
        loss_by_state = cdist(self.mu, new_Y, "euclidean").T**2
        V = loss_by_state.copy()

        s = np.zeros(n_obs, dtype=int)
        for t in range(n_obs - 1, 0, -1):
            V[t - 1] = loss_by_state[t - 1] + (V[t] + self.Gamma).min(axis=1)

        s[0] = V[0].argmin()
        for t in range(1, n_obs):
            s[t] = (self.Gamma[s[t - 1]] + V[t]).argmin()

        return s

    def _init_states(self):
        n_obs, n_features = self.Y.shape
        centers = np.zeros((self.n_states, n_features))
        center_idx = np.random.randint(n_obs)
        centers[0] = self.Y[center_idx]
        n_local_trials = 2 + int(np.log(self.n_states))
        closest_dist_sq = cdist(centers[0, None], self.Y, "euclidean")**2
        current_pot = closest_dist_sq.sum()

        for i in range(1, self.n_states):
            rand_vals = np.random.sample(n_local_trials) * current_pot
            candidate_ids = np.searchsorted(np.cumsum(closest_dist_sq), rand_vals)
            distance_to_candidates = cdist(self.Y[candidate_ids], self.Y, "euclidean")**2

            best_candidate = None
            best_pot = None
            best_dist_sq = None

            for trial in range(n_local_trials):
                new_dist_sq = np.minimum(closest_dist_sq, distance_to_candidates[trial])
                new_pot = new_dist_sq.sum()

                if best_candidate is None or new_pot < best_pot:
                    best_candidate = candidate_ids[trial]
                    best_pot = new_pot
                    best_dist_sq = new_dist_sq

            centers[i] = self.Y[best_candidate]
            current_pot = best_pot
            closest_dist_sq = best_dist_sq

        states = cdist(centers, self.Y, "euclidean").argmin(axis=0)
        return states

    def _m_step(self, mu):
        for i in np.unique(self.s):
            mu[i] = np.mean(self.Y[self.s == i], axis=0)

    def _e_step(self, mu, loss_old):
        loss_by_state = cdist(mu, self.Y, "euclidean").T**2
        V = loss_by_state.copy()

        for t in range(self.n_obs - 1, 0, -1):
            V[t - 1] = loss_by_state[t - 1] + (V[t] + self.Gamma).min(axis=1)

        self.s[0] = V[0].argmin()
        for t in range(1, self.n_obs):
            self.s[t] = (self.Gamma[self.s[t - 1]] + V[t]).argmin()

        loss = min(V[0])

        if self.tol:
            epsilon = loss_old - loss
            converged = epsilon < self.tol
        else:
            converged = np.array_equal(self.s, self.best_s)

        return loss, converged
