import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from .feature_engineer import apply_feature_engineering

class TimeSeriesWindowDataset(Dataset):
    """
    Pack a sequence of shape (T, D) into non-overlapping windows of (window_size, D),
    where each window corresponds to a slice of hidden states.
    """

    def __init__(self, X, S, window_size):
        """
        Args:
            X (np.ndarray): shape (T, D)
            S (np.ndarray): shape (T, )
            window_size (int)
        """
        super().__init__()
        self.X = X
        self.S = S
        self.window_size = window_size

        T_total = len(X)
        self.num_windows = T_total // window_size
        self.indices = [(i * window_size, (i + 1) * window_size) for i in range(self.num_windows)]

    def __len__(self):
        return self.num_windows

    def __getitem__(self, idx):
        start, end = self.indices[idx]
        x_win = self.X[start:end]  # (window_size, D)
        s_win = self.S[start:end]  # (window_size, )
        # Convert to torch.Tensor
        x_win = torch.tensor(x_win, dtype=torch.float32)
        s_win = torch.tensor(s_win, dtype=torch.long)
        return x_win, s_win


def standardize_data(X_train, X_val, X_test):
    """
    Standardize X based on the training data statistics.

    Args:
        X_train (np.ndarray): Training data.
        X_val (np.ndarray): Validation data.
        X_test (np.ndarray): Test data.

    Returns:
        tuple: Standardized X_train, X_val, X_test.
    """
    mean = X_train.mean(axis=0)
    std = X_train.std(axis=0)
    std[std == 0] = 1  # Avoid division by zero for constant features

    X_train = (X_train - mean) / std
    X_val = (X_val - mean) / std
    X_test = (X_test - mean) / std

    return X_train, X_val, X_test


def create_datasets(X, S, window_size=1000, train_ratio=0.6, val_ratio=0.2, train_len=None, val_len=None, standardize=False):
    """
    Split (X, S) into 6:2:2 ratios, optionally standardize X, 
    then create non-overlapping windows, and return using DataLoader.

    Args:
        X (np.ndarray): Input data.
        S (np.ndarray): Labels.
        window_size (int): Window size.
        train_ratio (float): Ratio of training data.
        val_ratio (float): Ratio of validation data.
        standardize (bool): Whether to standardize the data.

    Returns:
        tuple: train_dataset, val_dataset, test_dataset
    """
    T = len(X)
    # Split indices
    if train_len is None or val_len is None:
        train_end = int(train_ratio * T)
        val_end = int((train_ratio + val_ratio) * T)
    else:
        train_end = train_len
        val_end = train_len + val_len

    X_train, S_train = X[:train_end], S[:train_end]
    X_val, S_val = X[train_end:val_end], S[train_end:val_end]
    X_test, S_test = X[val_end:], S[val_end:]

    if standardize:
        X_train, X_val, X_test = standardize_data(X_train, X_val, X_test)

    train_dataset = TimeSeriesWindowDataset(X_train, S_train, window_size)
    val_dataset = TimeSeriesWindowDataset(X_val, S_val, window_size)
    test_dataset = TimeSeriesWindowDataset(X_test, S_test, window_size)

    return train_dataset, val_dataset, test_dataset


def create_dataloaders(X, S, window_size=1000, train_ratio=0.6, val_ratio=0.2, train_len=None, val_len=None, batch_size=16, standardize=False, feature_engineer=False, train_shuffle=True):
    """
    Create DataLoaders for train, validation, and test datasets.

    Args:
        X (np.ndarray): Input data.
        S (np.ndarray): Labels.
        window_size (int): Window size.
        train_ratio (float): Ratio of training data.
        val_ratio (float): Ratio of validation data.
        batch_size (int): Batch size.
        standardize (bool): Whether to standardize the data.

    Returns:
        tuple: train_loader, val_loader, test_loader
    """
    # Feature engineering
    if feature_engineer:
        X = apply_feature_engineering(X)
    
    train_dataset, val_dataset, test_dataset = create_datasets(X, S, window_size, train_ratio, val_ratio, train_len, val_len, standardize)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=train_shuffle)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    return train_loader, val_loader, test_loader
