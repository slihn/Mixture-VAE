import numpy as np
from scipy.spatial.distance import cdist

class JumpModelBatch:
    def __init__(self, n_states, jump_penalty=1e-5, max_iter=10, n_init=10, tol=None, verbose=False):
        """
        n_states: Number of states
        jump_penalty: Jump penalty coefficient
        max_iter: Number of EM iterations
        n_init: Number of random initializations to find the best result
        tol: Convergence threshold
        verbose: Whether to print intermediate information
        """
        self.n_states = n_states
        self.jump_penalty = jump_penalty
        self.max_iter = max_iter
        self.n_init = n_init
        self.tol = tol
        self.verbose = verbose

        # Gamma: State transition/jump penalty matrix, main diagonal is 0, off-diagonal elements are jump_penalty
        self.Gamma = jump_penalty * (1 - np.eye(n_states))

        self.mu = None  # Shape (n_states, D)
        self.best_loss = None
        self.best_s = None

    def fit(self, Y, initial_states=None):
        """
        Y: Data of shape (B, T, D), B sequences, each of length T, with D-dimensional features at each time
        initial_states: Initial allocation of shape (B, T), optional; if not provided, random initialization is used
        """
        # B, T, D
        self.B, self.T, self.D = Y.shape
        self.Y = Y

        # Initialize s: (B, T) where each position is a state [0..n_states-1]
        if initial_states is not None:
            s = np.asarray(initial_states, dtype=np.int64)
            # Use the initial allocation if its state categories match n_states; otherwise, randomize
            if len(np.unique(s)) == self.n_states:
                self.s = s.copy()
            else:
                self.s = self._init_states()
        else:
            self.s = self._init_states()

        # Record the best results
        self.best_loss = None
        self.best_s = None

        for init in range(self.n_init):
            # Randomly initialize mu
            mu = np.zeros((self.n_states, self.D))

            loss_old = 1e10
            for it in range(self.max_iter):
                # M step: Update mu
                self._m_step(mu)
                # E step: Update s and return the loss value
                loss, converged = self._e_step(mu, loss_old)

                if self.verbose:
                    print(f"[init {init:2d}, iter {it:2d}] loss = {loss:.6f}")

                if converged:
                    break
                loss_old = loss

            # Update the best results if better
            if self.best_loss is None or loss_old < self.best_loss:
                self.best_loss = loss_old
                self.best_s = self.s.copy()
                self.mu = mu.copy()

            # Randomly reinitialize s for each n_init
            self.s = self._init_states()

        # Use the best s for the final model
        self.s = self.best_s

    def inference(self, new_Y):
        """
        Perform inference on new data new_Y (shape: (B, T, D)), returning s_hat (B, T)
        """
        if self.mu is None:
            raise ValueError("Model parameters are not fitted. Call fit() before inference.")

        Bn, Tn, Dn = new_Y.shape
        if Dn != self.D:
            raise ValueError(f"new_Y feature dim {Dn} != model feature dim {self.D}.")

        # Calculate cost_by_state: (Bn, Tn, n_states)
        # Use broadcasting to manually compute (Y - mu)^2, or use cdist for small data
        # scipy cdist is less convenient for processing Bn sequences at once, so we use broadcasting:
        # new_Y shape: (Bn, Tn, D)
        # mu shape:     (n_states, D)
        # => Expand to: (1,    1,    n_states, D) vs. (Bn, Tn, 1, D)
        # => Result: (Bn, Tn, n_states, D)
        # => Sum along axis=-1
        diff = new_Y[:, :, None, :] - self.mu[None, None, :, :]
        cost_by_state = np.sum(diff**2, axis=-1)  # (Bn, Tn, n_states)

        # Backward dynamic programming: V[b, t, i] = cost_by_state[b, t, i] + min_j [ V[b, t+1, j] + Gamma[j, i] ]
        V = cost_by_state.copy()  # (Bn, Tn, n_states)

        # Update V backward
        for t in range(Tn - 2, -1, -1):
            # V[:, t+1, :] shape: (Bn, n_states)
            # Gamma shape: (n_states, n_states)
            # Compute min_j [ V[b, t+1, j] + Gamma[j, i] ] by minimizing over j
            # This can be done by broadcasting (Bn, 1, n_states) + (1, n_states, n_states), then min along axis=-1
            # Result: (Bn, n_states)
            tmp = V[:, t+1, :, None] + self.Gamma[None, :, :]  # shape (Bn, n_states, n_states)
            min_part = tmp.min(axis=1)  # shape (Bn, n_states)
            V[:, t, :] += min_part  # Broadcast to (Bn, n_states)

        # Backtrack s_hat
        s_hat = np.zeros((Bn, Tn), dtype=np.int64)
        # Optimal state at the first time step t=0
        s_hat[:, 0] = np.argmin(V[:, 0, :], axis=1)

        for t in range(1, Tn):
            # s_hat[:, t-1] shape: (Bn,)
            # Gamma[s_hat[:, t-1], :] shape: (Bn, n_states)
            # V[:, t, :] shape: (Bn, n_states)
            # => Add and argmin over n_states dimension
            prev_states = s_hat[:, t-1]  # (Bn,)
            cost_to_go = self.Gamma[prev_states, :] + V[:, t, :]
            s_hat[:, t] = np.argmin(cost_to_go, axis=1)

        return s_hat

    def _init_states(self):
        """
        Select n_states centers in (B*T, D) space in a k-means++ style,
        then assign (B*T) points to the nearest center to obtain initial s (B, T).
        """
        # Transform (B, T, D) into (B*T, D)
        Y_2d = self.Y.reshape(-1, self.D)  # shape: (B*T, D)
        n_total = Y_2d.shape[0]

        centers = np.zeros((self.n_states, self.D))
        center_idx = np.random.randint(n_total)
        centers[0] = Y_2d[center_idx]

        n_local_trials = 2 + int(np.log(self.n_states))
        closest_dist_sq = cdist(centers[0, None], Y_2d, "euclidean")**2
        current_pot = closest_dist_sq.sum()

        for i in range(1, self.n_states):
            rand_vals = np.random.sample(n_local_trials) * current_pot
            candidate_ids = np.searchsorted(np.cumsum(closest_dist_sq), rand_vals)
            distance_to_candidates = cdist(Y_2d[candidate_ids], Y_2d, "euclidean")**2

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

            centers[i] = Y_2d[best_candidate]
            current_pot = best_pot
            closest_dist_sq = best_dist_sq

        # Assign all (B*T) points to the nearest center
        states_1d = cdist(centers, Y_2d, "euclidean").argmin(axis=0)  # shape: (B*T,)
        # Reshape back to (B, T)
        states = states_1d.reshape(self.B, self.T)
        return states

    def _m_step(self, mu):
        """
        mu[i] = Mean of all samples assigned to state i
        """
        # Flatten Y, s to 1D for boolean indexing
        Y_2d = self.Y.reshape(-1, self.D)    # (B*T, D)
        s_1d = self.s.reshape(-1)           # (B*T,)

        # Update state by state
        for i in range(self.n_states):
            # Find all samples assigned to state i
            idx = (s_1d == i)
            if np.any(idx):
                mu[i] = Y_2d[idx].mean(axis=0)
            else:
                # If no samples are assigned to a state, randomly pick a point or leave unchanged
                mu[i] = Y_2d[np.random.randint(Y_2d.shape[0])]

    def _e_step(self, mu, loss_old):
        """
        Update each sample's state based on the new mu and return (loss, converged)
        """
        # Calculate cost_by_state: (B, T, n_states)
        diff = self.Y[:, :, None, :] - mu[None, None, :, :]
        cost_by_state = np.sum(diff**2, axis=-1)

        # Dynamic programming
        V = cost_by_state.copy()  # shape: (B, T, n_states)
        for t in range(self.T - 2, -1, -1):
            tmp = V[:, t+1, :, None] + self.Gamma[None, :, :]
            min_part = tmp.min(axis=1)
            V[:, t, :] += min_part

        # Backtracking
        s_new = np.zeros((self.B, self.T), dtype=int)
        s_new[:, 0] = np.argmin(V[:, 0, :], axis=1)
        for t in range(1, self.T):
            prev_states = s_new[:, t-1]
            cost_to_go = self.Gamma[prev_states, :] + V[:, t, :]
            s_new[:, t] = np.argmin(cost_to_go, axis=1)

        # Calculate the new loss
        loss = V[:, 0, :].min(axis=1).sum()  # Sum of the minimum values of B sequences

        self.s = s_new

        if self.tol is not None:
            epsilon = loss_old - loss
            converged = (epsilon < self.tol)
        else:
            # Consider converged if allocation is identical to the previous one
            converged = False
            # converged = np.array_equal(self.s, self.best_s)

        return loss, converged
