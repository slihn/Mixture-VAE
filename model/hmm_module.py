from hmmlearn.hmm import GaussianHMM
import numpy as np


class GaussianHMMModule:
    """
    Using hmmlearn.hmm.GaussianHMM
    """
    def __init__(self, n_components=2, covariance_type='full', n_iter=10, random_state=42):
        self.model = GaussianHMM(
            n_components=n_components, 
            covariance_type=covariance_type, 
            n_iter=n_iter,
            random_state=random_state
        )

    def fit(self, train_loader):
        """
        Aggregate all windows from the training set and train using fit(X, lengths)
        """
        X_all = []
        lengths = []
        for x_batch, _ in train_loader:
            # x_batch: (batch_size, window_size, D)
            batch_size, window_size, D = x_batch.shape
            # Append each sample (window) to X_all
            # lengths stores the length of each window for hmmlearn
            for i in range(batch_size):
                window_np = x_batch[i].numpy()  # (window_size, D)
                X_all.append(window_np)
                lengths.append(window_size)

        # Concatenate to (total_time, D)
        X_all = np.concatenate(X_all, axis=0)  # Shape: (N, D)
        self.model.fit(X_all, lengths)

    def inference(self, test_loader):
        """
        Predict the most likely hidden state sequence for each window in the test set,
        and collect all true S and predicted S.
        """
        all_true_s = []
        all_pred_s = []

        for x_batch, s_batch in test_loader:
            batch_size, window_size, D = x_batch.shape
            # Predict for each window
            for i in range(batch_size):
                x_win = x_batch[i].numpy()  # (window_size, D)
                s_win = s_batch[i].numpy()  # (window_size, )
                # hmmlearn predict returns Viterbi path
                pred_states = self.model.predict(x_win)
                all_true_s.append(s_win)
                all_pred_s.append(pred_states)

        # Concatenate to 1D arrays
        all_true_s = np.stack(all_true_s, axis=0)
        all_pred_s = np.stack(all_pred_s, axis=0)
        return all_true_s, all_pred_s