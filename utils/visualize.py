import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
import seaborn as sns
import os

def plot_clusters(data, 
                  cluster_labels, 
                  cluster_count=None, 
                  colors=None, 
                  label_names=None, 
                  pca_components=2, 
                  title=None, 
                  save_path=None, 
                  dpi=100, 
                  marker_size=1,
                  legend=True):
    """
    General function to visualize clusters.

    If the input data has only one feature, a density plot (KDE) is generated for each category.
    Otherwise (i.e., for two or more features), PCA (if necessary) is applied and a scatter plot is drawn.

    Parameters:
        data: High-dimensional data as a numpy array with shape (n_samples, n_features).
              If a PyTorch Tensor is provided, it will be automatically converted to a numpy array.
        cluster_labels: Array-like of cluster labels corresponding to each data point.
        cluster_count: Number of clusters. If None, the number of unique values in cluster_labels is used.
        colors: List of colors for each cluster (e.g., ['red', 'blue', 'green']).
                If None, the default matplotlib 'tab10' colormap is used.
        label_names: List of names for each cluster. If None, the unique cluster labels are used.
        pca_components: Number of PCA components to compute (default is 2).
                        Only the first two components are used for plotting.
        title: Title for the plot.
        save_path: File path to save the plot. If None or empty, the plot is displayed.
        dpi: Resolution of the plot.
        marker_size: Size of the markers in the scatter plot.

    Returns:
        None. The plot is either saved to the provided path or displayed.
    """
    # Convert data to a numpy array if it's a PyTorch Tensor
    if not isinstance(data, np.ndarray):
        try:
            import torch
            if isinstance(data, torch.Tensor):
                data = data.cpu().detach().numpy()
        except ImportError:
            pass

    # Ensure cluster_labels is a one-dimensional numpy array
    cluster_labels = np.asarray(cluster_labels).flatten()

    # Determine the number of clusters if not provided
    if cluster_count is None:
        unique_labels = np.unique(cluster_labels)
        cluster_count = len(unique_labels)
    else:
        unique_labels = np.arange(cluster_count)

    # --- New 1D Data Functionality ---
    # If the input data has only one feature, create a density (KDE) plot for each category.
    if data.shape[1] == 1:
        X_1D = data[:, 0]
        plt.figure(figsize=(8, 6), dpi=dpi)
        for label in np.unique(cluster_labels):
            subset = X_1D[cluster_labels == label]
            sns.kdeplot(subset, label=f"Category {label}", fill=True, alpha=0.5, linewidth=2)
        plt.xlabel("Value of Feature")
        plt.ylabel("Density")
        if title:
            plt.title(title)
        if legend:
            plt.legend()
        if save_path:
            plt.savefig(save_path)
            plt.close()
        else:
            plt.show()
        return
    # --- End of 1D functionality ---

    # For 2D or higher-dimensional data, use the original scatter plot code.
    # Use default colors if none provided.
    if colors is None:
        cmap = plt.get_cmap('tab10')
        colors = [cmap(i) for i in range(cluster_count)]
    else:
        if len(colors) < cluster_count:
            raise ValueError("The number of provided colors is less than the number of clusters.")

    # Use default label names if not provided.
    if label_names is None:
        label_names = [str(label) for label in unique_labels]
    else:
        if len(label_names) < cluster_count:
            raise ValueError("The number of provided label names is less than the number of clusters.")

    # Create a mapping from each label to a corresponding color.
    label_to_color = {label: colors[idx] for idx, label in enumerate(unique_labels)}
    point_colors = [label_to_color[label] for label in cluster_labels]

    # Apply PCA if data has more than 2 features.
    if data.shape[1] > 2:
        pca = PCA(n_components=pca_components)
        data_pca = pca.fit_transform(data)
        x = data_pca[:, 0]
        y = data_pca[:, 1]
    elif data.shape[1] == 2:
        x = data[:, 0]
        y = data[:, 1]
    else:
        # This branch should not be reached because 1D data is handled above.
        raise ValueError("Unexpected data dimensionality.")

    plt.figure(dpi=dpi)
    plt.scatter(x, y, color=point_colors, s=marker_size)

    # Create a legend by plotting empty scatter plots.
    for idx in range(cluster_count):
        plt.scatter([], [], color=colors[idx], label=label_names[idx])

    if title:
        plt.title(title)
    if legend:
        plt.legend()

    if save_path:
        save_dir = os.path.dirname(save_path)
        os.makedirs(save_dir, exist_ok=True)
        plt.savefig(save_path)
    plt.show()
