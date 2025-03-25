from sklearn.cluster import KMeans
import numpy as np

class KMeansModule:
    """
    A KMeans module using sklearn.cluster.KMeans with k-means++ initialization by default.
    """
    def __init__(self, n_clusters=2, n_init=10, max_iter=300, random_state=42):
        """
        Initialize the KMeans model with the given hyperparameters.
        
        :param n_clusters: Number of clusters
        :param n_init: Number of times the algorithm will be run with different centroid seeds
        :param max_iter: Maximum number of iterations for each run
        :param random_state: Determines random number generation for centroid initialization
        """
        self.model = KMeans(
            n_clusters=n_clusters,
            n_init=n_init,
            max_iter=max_iter,
            random_state=random_state
        )

    def fit(self, train_loader):
        """
        Aggregate all windows from the training set into a single array of shape (N, D)
        and fit the KMeans model.
        
        :param train_loader: A data loader that yields (x_batch, y_batch) tuples
                             where x_batch is of shape (batch_size, window_size, D)
        """
        X_all = []
        for x_batch, _ in train_loader:
            # x_batch has shape (batch_size, window_size, D)
            batch_size, window_size, D = x_batch.shape

            # Append each sample (window) to X_all
            for i in range(batch_size):
                # Convert from torch.Tensor to numpy.ndarray
                window_np = x_batch[i].numpy()  # shape: (window_size, D)
                X_all.append(window_np)

        # Concatenate all windows along the time dimension (window_size) -> (N, D)
        X_all = np.concatenate(X_all, axis=0)
        self.model.fit(X_all)

    def inference(self, test_loader):
        """
        Predict cluster labels for each window in the test set. 
        Collects all true labels (if available) and predicted labels.
        
        :param test_loader: A data loader that yields (x_batch, s_batch) tuples
                            where x_batch is (batch_size, window_size, D) and
                            s_batch is (batch_size, window_size)
        :return: A tuple (all_true_s, all_pred_s), each of shape (number_of_windows, window_size)
        """
        all_true_s = []
        all_pred_s = []

        for x_batch, s_batch in test_loader:
            batch_size, window_size, D = x_batch.shape
            # Predict labels for each window
            for i in range(batch_size):
                x_win = x_batch[i].numpy()  # shape: (window_size, D)
                s_win = s_batch[i].numpy()  # shape: (window_size, )
                pred_labels = self.model.predict(x_win)  # shape: (window_size, )
                
                all_true_s.append(s_win)
                all_pred_s.append(pred_labels)

        # Stack all arrays into (num_windows, window_size)
        all_true_s = np.stack(all_true_s, axis=0)
        all_pred_s = np.stack(all_pred_s, axis=0)
        return all_true_s, all_pred_s
