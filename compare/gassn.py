"""Compare clustering models on synthetic GAS-SN regime data, scored by balanced accuracy."""

import numpy as np
import pandas as pd
import torch
from scipy.stats import kurtosis, skew

from data_code.synthetic_data import generate_hmm_data
from data_code.dataloader import create_dataloaders
from model.vae_module import VAEModule
from model.jump_module import JumpModule
from model.kmeans_module import KMeansModule
from model.hmm_module import GaussianHMMModule
from utils.metrics import balanced_accuracy


class VAEParams:
    """Flat config namespace for MixtureVAE.

    MixtureVAE reads its sub-net settings as getattr(args, f"{module}_type") and friends,
    so the names must stay flat -- do not nest them into per-module dicts.
    """
    name = 'mixture_vae'

    # backbone
    tau = 1.0
    hard = False
    transition = 'jump'
    lamda_m = 4.0
    lamda_i = 4.0
    lamda_t = 4.0
    hidden_dim = 16
    loss_mode = 'sum'

    # s_x
    s_x_type = 'lstm'
    s_clamp = 5
    s_x_dropout = 0.1
    s_x_lstm_hidden = 64
    s_x_lstm_layers = 1

    # z_sx (q)
    z_sx_type = 'mlp'
    z_sx_dropout = 0.1
    z_sx_hiddens = [128, 128]

    # reconstruction: if True use x_sz, else x_z
    reconstruction_on_s = True
    reconstruction_on_z = 'p'

    # x_sz
    x_sz_type = 'mlp'
    x_sz_dropout = 0.1
    x_sz_hiddens = [128, 128]

    # x_z
    x_z_type = 'mlp'
    x_z_dropout = 0.1
    x_z_hiddens = [128, 128]

    def __init__(self, feature, n_cluster, seq_len, loss_clamp, **overrides):
        # derived from the data, so they cannot be class-level defaults
        self.feature = feature
        self.n_cluster = n_cluster
        self.seq_len = seq_len
        self.loss_clamp = loss_clamp

        for key, value in overrides.items():
            if not hasattr(self, key):
                raise AttributeError(f"unknown VAE param: {key}")
            setattr(self, key, value)


class GAS_SN_Comparator:
    """Generate a 2-state HMM with GAS-SN emissions and score models by balanced accuracy.

    The GAS-SN path in generate_hmm_data is univariate only, hence D == 1.

        cmp = GAS_SN_Comparator()
        cmp.compare()               # -> DataFrame of balanced accuracy per model
    """
    MODELS = ('vae', 'jump', 'kmeans', 'hmm')

    def __init__(self,
                 T=100008, D=1, num_states=2,
                 alpha=1.1, k=2.75, beta=0.0, loc=0.001, scale=0.003,
                 stay_prob=0.96, clip_factor=12.0, 
                 chunk_size=1000, seed=42,
                 window_size=500, batch_size=32, train_ratio=0.6, val_ratio=0.2,
                 vae_lr=1e-3, vae_epochs=500,
                 jump_penalty=100.0, jump_max_iter=100,
                 kmeans_n_init=10, kmeans_max_iter=300,
                 hmm_n_iter=100, hmm_covariance_type='full',
                 **vae_params):
        assert D == 1, "GAS_SN emissions are univariate, so D must be 1"

        self.T = T
        self.D = D
        self.num_states = num_states

        # alpha=1 recovers student's t; beta=0 is unskewed
        self.alpha = alpha
        self.k = k
        self.beta = beta
        self.loc = loc
        self.scale = scale   # the GAS_SN scale parameter, NOT the realized standard deviation

        self.stay_prob = stay_prob
        self.clip_factor = clip_factor  # gas-sn tolerates a larger clip factor than t
        self.chunk_size = chunk_size
        # an int makes S and X both reproducible; None gives fresh randomness.
        # Vary it across replicates (seed=rep) to get independent draws.
        self.seed = seed

        self.window_size = window_size
        self.batch_size = batch_size
        self.train_ratio = train_ratio
        self.val_ratio = val_ratio

        self.vae_lr = vae_lr
        self.vae_epochs = vae_epochs
        self.jump_penalty = jump_penalty
        self.jump_max_iter = jump_max_iter
        self.kmeans_n_init = kmeans_n_init
        self.kmeans_max_iter = kmeans_max_iter
        self.hmm_n_iter = hmm_n_iter
        self.hmm_covariance_type = hmm_covariance_type
        self.vae_params_overrides = vae_params

        self.S = None
        self.X = None
        self.train_loader = None
        self.val_loader = None
        self.test_loader = None
        self.modules = {}
        self.results = {}

    # ------------------------------------------------------------------ data
    @property
    def startprob(self):
        return np.full(self.num_states, 1.0 / self.num_states)

    @property
    def transition_probs(self):
        jump = (1.0 - self.stay_prob) / (self.num_states - 1)
        mx = np.full((self.num_states, self.num_states), jump)
        np.fill_diagonal(mx, self.stay_prob)
        return mx

    @property
    def emission_params(self):
        """One GAS-SN per state, straddling zero at +/- loc."""
        signs = np.linspace(-1.0, 1.0, self.num_states)
        return [
            {'alpha': self.alpha, 'k': self.k, 'beta': self.beta,
             'loc': sign * self.loc, 'scale': self.scale,
             'shape': np.eye(self.D) * self.scale ** 2}
            for sign in signs
        ]

    def generate(self):
        """Sample the hidden states S and observations X."""
        self.S, self.X = generate_hmm_data(
            T=self.T,
            D=self.D,
            num_states=self.num_states,
            startprob=self.startprob,
            transition_probs=self.transition_probs,
            emission_dist='gassn',
            hmm_params=self.emission_params,
            seed=self.seed,
            clip_factor=self.clip_factor,
            chunk_size=self.chunk_size,
        )
        return self.S, self.X

    def stats(self):
        """Moments overall and per state -- kurtosis is damped by clip_factor."""
        if self.X is None:
            self.generate()

        rows = [('all', self.X)]
        rows += [(state, self.X[self.S == state]) for state in np.unique(self.S)]
        return pd.DataFrame(
            [{'state': name,
              'n': len(x),
              'mean': np.mean(x),
              'std': np.std(x),
              'skew': float(np.ravel(skew(x))[0]),
              'kurtosis': float(np.ravel(kurtosis(x))[0])}
             for name, x in rows]
        ).set_index('state')

    def dataloaders(self):
        if self.train_loader is None:
            if self.X is None:
                self.generate()
            # A seeded generator pins the train loader's shuffle permutation. Seeding torch's
            # global RNG is not sufficient: RandomSampler redraws on every iteration.
            generator = None
            if self.seed is not None:
                generator = torch.Generator().manual_seed(int(self.seed))
            self.train_loader, self.val_loader, self.test_loader = create_dataloaders(
                self.X, self.S,
                window_size=self.window_size,
                train_ratio=self.train_ratio,
                val_ratio=self.val_ratio,
                batch_size=self.batch_size,
                standardize=True,
                feature_engineer=True,
                generator=generator,
            )
        return self.train_loader, self.val_loader, self.test_loader

    # ---------------------------------------------------------------- models
    @property
    def vae_params(self):
        return VAEParams(
            feature=self.D * 15,  # width produced by apply_feature_engineering
            n_cluster=self.num_states,
            seq_len=self.window_size,
            loss_clamp=self.batch_size * self.window_size * 10,
            **self.vae_params_overrides,
        )

    def _score(self, name, module, **fit_kwargs):
        """Fit on train, predict on test, record balanced accuracy."""
        train_loader, _, test_loader = self.dataloaders()
        module.fit(train_loader, **fit_kwargs)
        true_s, pred_s = module.inference(test_loader)

        bac = balanced_accuracy(true_s.ravel(), pred_s.ravel(), n_classes=self.num_states)
        self.modules[name] = module
        self.results[name] = bac
        return bac

    def _seed_torch(self):
        """Pin torch's global RNG. Must run BEFORE a module is constructed, because weight
        init draws from it -- and a module built in a call expression is created before the
        callee's body runs."""
        if self.seed is not None:
            torch.manual_seed(int(self.seed))

    def fit_vae(self):
        self._seed_torch()          # covers the VAE's weight init
        return self._score('vae', VAEModule(self.vae_params),
                           lr=self.vae_lr, epochs=self.vae_epochs)

    def fit_jump(self):
        return self._score('jump', JumpModule(self.num_states,
                                              jump_penalty=self.jump_penalty,
                                              max_iter=self.jump_max_iter))

    def fit_kmeans(self):
        return self._score('kmeans', KMeansModule(n_clusters=self.num_states,
                                                  n_init=self.kmeans_n_init,
                                                  max_iter=self.kmeans_max_iter))

    def fit_hmm(self):
        return self._score('hmm', GaussianHMMModule(n_components=self.num_states,
                                                    covariance_type=self.hmm_covariance_type,
                                                    n_iter=self.hmm_n_iter,
                                                    random_state=self.seed))

    def compare(self, models=MODELS, verbose=True):
        """Fit each model and return its balanced accuracy on the test set."""
        for name in models:
            bac = getattr(self, f"fit_{name}")()
            if verbose:
                print(f"[{name}] Balanced Accuracy: {bac:.4f}")

        return (pd.DataFrame({'balanced_accuracy': self.results})
                .rename_axis('model')
                .loc[list(models)])
