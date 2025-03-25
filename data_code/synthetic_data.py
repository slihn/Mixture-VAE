import os
import numpy as np
from scipy.stats import multivariate_t
from hmmlearn import hmm

def generate_hmm_data(
    T=100000,
    D=3,
    num_states=2,
    startprob=None,
    transition_probs=None,
    emission_dist='gaussian',
    hmm_params=None,
    seed=42,
    clip_factor=None,
    chunk_size=1000,
    save_path=None
):
    """
    Generate time series data for an HMM with flexible emission distributions:
      - S: shape (T,), hidden state sequence
      - X: shape (T, D), observed sequence

    Parameters:
        T: int
            Length of the time series.
        D: int
            Dimensionality of the observations.
        num_states: int
            Number of hidden states.
        startprob: np.array of shape (num_states,)
            Initial state distribution. If None, a uniform distribution is used.
        transition_probs: np.array of shape (num_states, num_states)
            State transition matrix. If None, a random transition matrix is generated.
        emission_dist: str, either 't' or 'gaussian'
            Type of emission distribution.
        hmm_params:
            If 't': a list of dictionaries (length = num_states):
                [
                    {
                      'df':    <degrees_of_freedom>,
                      'loc':   <loc_vector>,
                      'shape': <shape_matrix_or_vector>
                    },
                    ...
                ]
            If 'gaussian': a dictionary with the following keys:
                {
                    'means': shape = (num_states, D),
                    'covs':  shape = (num_states, D, D)
                }
        seed: int
            Random seed.
        clip_factor: float or None
            If not None and > 0, samples out of the range
                [loc - k*std, loc + k*std]
            (or [mean - k*std, mean + k*std]) will be discarded and re-sampled.
        chunk_size: int
            How many samples to draw in one batch during rejection sampling.

    Returns:
        S: np.array of shape (T,)
            Hidden state sequence.
        X: np.array of shape (T, D)
            Observed data sequence.
    """
    np.random.seed(seed)

    # If transition_probs is not provided, generate a random one
    if transition_probs is None:
        transition_probs = np.random.dirichlet(alpha=[1] * num_states, size=num_states)
    
    # If startprob is not provided, use a uniform distribution
    if startprob is None:
        startprob = np.ones(num_states) / num_states

    # -------------------
    # 1) Generate hidden states S
    # -------------------
    if emission_dist == 't':
        # Use CategoricalHMM just to generate a state sequence
        model = hmm.CategoricalHMM(n_components=num_states)
        model.startprob_ = startprob
        model.transmat_ = transition_probs
        model.emissionprob_ = np.eye(num_states)
        S = model.sample(T)[0].flatten()

    elif emission_dist == 'gaussian':
        # Use GaussianHMM from hmmlearn
        model = hmm.GaussianHMM(n_components=num_states, covariance_type="full", random_state=seed)
        model.startprob_ = startprob
        model.transmat_ = transition_probs
        
        if hmm_params is not None:
            model.means_ = (hmm_params['means']
                            if hmm_params['means'] is not None
                            else np.random.randn(num_states, D))
            model.covars_ = (hmm_params['covs']
                             if hmm_params['covs'] is not None
                             else np.array([np.eye(D) for _ in range(num_states)]))
        else:
            model.means_ = np.random.randn(num_states, D)
            model.covars_ = np.array([np.eye(D) for _ in range(num_states)])

        # We only need the state sequence here (S)
        # So let's sample with length T, but we won't use the X from here
        _, S = model.sample(T)
    else:
        raise ValueError("Unsupported emission_dist. Choose either 't' or 'gaussian'.")

    # -------------------
    # 2) Generate observations X
    # -------------------
    X = np.zeros((T, D))

    if emission_dist == 't':
        # Prepare distributions
        t_dists = [
            multivariate_t(df=params['df'], loc=params['loc'], shape=params['shape'])
            for params in hmm_params
        ]

        # For each state, generate exactly the needed number of samples with rejection
        for state in range(num_states):
            n_required = np.sum(S == state)
            if n_required == 0:
                continue

            # If no clip_factor or <= 0, just standard sampling
            if not clip_factor or clip_factor <= 0:
                samples = t_dists[state].rvs(size=n_required)
                if D == 1:
                    samples = samples.reshape(-1, 1)
                X[S == state] = samples
            else:
                # Rejection sampling to keep only samples within the bounding region
                loc_ = hmm_params[state]['loc']
                shape_ = hmm_params[state]['shape']
                
                if shape_.ndim == 1:
                    lower = loc_ - clip_factor * shape_
                    upper = loc_ + clip_factor * shape_
                else:
                    diag_std = np.sqrt(np.diag(shape_))
                    lower = loc_ - clip_factor * diag_std
                    upper = loc_ + clip_factor * diag_std
                
                X[S == state] = _rejection_sample(
                    distribution=t_dists[state],
                    n_samples=n_required,
                    D=D,
                    lower=lower,
                    upper=upper,
                    chunk_size=chunk_size
                )

    elif emission_dist == 'gaussian':
        means_ = model.means_
        covs_ = model.covars_
        
        for state in range(num_states):
            n_required = np.sum(S == state)
            if n_required == 0:
                continue

            # If no clip_factor or <= 0, just standard sampling
            if not clip_factor or clip_factor <= 0:
                # Sample from a normal distribution with means_[state], covs_[state]
                # We can just use np.random.multivariate_normal for that
                samples = np.random.multivariate_normal(means_[state], covs_[state], size=n_required)
                X[S == state] = samples
            else:
                # Rejection sampling approach
                mean_ = means_[state]
                diag_std = np.sqrt(np.diag(covs_[state]))
                
                lower = mean_ - clip_factor * diag_std
                upper = mean_ + clip_factor * diag_std

                # Create a callable distribution for convenience
                # We'll do a small helper that samples from MVN.
                def _gaussian_rvs(size):
                    return np.random.multivariate_normal(mean_, covs_[state], size=size)
                
                X[S == state] = _rejection_sample(
                    distribution=_gaussian_rvs,
                    n_samples=n_required,
                    D=D,
                    lower=lower,
                    upper=upper,
                    chunk_size=chunk_size
                )
    
    if save_path is not None:
        folder_path = f"../data/synthetic/{emission_dist}-hmm/{save_path}"
        s_filename = f"{folder_path}/S_data.npy"
        x_filename = f"{folder_path}/X_data.npy"

        os.makedirs(folder_path, exist_ok=True)
        np.save(s_filename, S)
        np.save(x_filename, X)

    return S, X


def _rejection_sample(distribution, n_samples, D, lower, upper, chunk_size=1000):
    """
    A helper function to perform rejection sampling. It keeps drawing from the given
    distribution in chunks, discards samples outside [lower, upper] for each dimension,
    and accumulates valid samples until n_samples is reached.

    Parameters:
    ----------
    distribution : callable or scipy.stats object
        If it's a scipy.stats distribution, it should have an rvs(size) method.
        If it's a callable, it should return an array of shape (chunk_size, D).
    n_samples : int
        Number of valid samples required.
    D : int
        Dimensionality.
    lower : array-like of shape (D,)
        Lower bound for each dimension.
    upper : array-like of shape (D,)
        Upper bound for each dimension.
    chunk_size : int
        How many samples to generate at once.

    Returns:
    -------
    samples : np.array of shape (n_samples, D)
        Valid samples that fall within the specified bounds.
    """
    valid_samples = []

    # Check if distribution is a scipy.stats object or a callable
    is_scipy_dist = hasattr(distribution, 'rvs')

    while len(valid_samples) < n_samples:
        # Generate a batch
        if is_scipy_dist:
            batch = distribution.rvs(size=chunk_size)
        else:
            batch = distribution(chunk_size)

        # Ensure shape is (chunk_size, D) if D=1
        if D == 1 and batch.ndim == 1:
            batch = batch.reshape(-1, 1)

        # Apply bounds
        mask = np.all((batch >= lower) & (batch <= upper), axis=1)
        valid = batch[mask]
        valid_samples.append(valid)

    # Concatenate and keep only the first n_samples
    valid_samples = np.concatenate(valid_samples, axis=0)[:n_samples]
    return valid_samples


if __name__ == "__main__":
    # Example usage:
    # 1) T-distribution, no rejection (clip_factor=None):
    S_t, X_t = generate_hmm_data(
        T=1000,
        D=2,
        num_states=2,
        emission_dist='t',
        hmm_params=[
            {'df': 5, 'loc': np.array([0., 0.]), 'shape': np.eye(2)},
            {'df': 5, 'loc': np.array([5., 5.]), 'shape': 2 * np.eye(2)},
        ],
        clip_factor=None  # no rejection
    )
    print("T-dist sample (no rejection), X_t shape =", X_t.shape)

    # 2) T-distribution, with rejection in range [loc ± 3*std_diag]:
    S_t2, X_t2 = generate_hmm_data(
        T=1000,
        D=2,
        num_states=2,
        emission_dist='t',
        hmm_params=[
            {'df': 5, 'loc': np.array([0., 0.]), 'shape': np.eye(2)},
            {'df': 5, 'loc': np.array([5., 5.]), 'shape': 2 * np.eye(2)},
        ],
        clip_factor=3.0  # keep only samples in loc ± 3*sqrt(diag(shape))
    )
    print("T-dist sample (with rejection), X_t2 shape =", X_t2.shape)

    # 3) Gaussian distribution, with rejection:
    S_g, X_g = generate_hmm_data(
        T=1000,
        D=2,
        num_states=2,
        emission_dist='gaussian',
        hmm_params={
            'means': np.array([[0., 0.], [5., 5.]]),
            'covs': np.array([np.eye(2), 2 * np.eye(2)])
        },
        clip_factor=2.0  # keep only samples in mean ± 2*std_diag
    )
    print("Gaussian sample (with rejection), X_g shape =", X_g.shape)
