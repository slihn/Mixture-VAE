from itertools import permutations
from sklearn.metrics import balanced_accuracy_score, roc_auc_score
import numpy as np

def balanced_accuracy(y_true, y_pred, n_classes=2):
    """
    Calculates the balanced accuracy for multi-class classification with label alignment module.

    Args:
        y_true (array-like of shape (n_samples,)): True labels of the samples.
        y_pred (array-like of shape (n_samples,)): Predicted labels of the samples.

    Returns:
        float: The balanced accuracy score.
    """
    max_score = float('-inf')
    for perm in permutations(range(n_classes)):
        y_perm = [perm[y] for y in y_pred]
        score = balanced_accuracy_score(y_true, y_perm)
        if score > max_score:
            max_score = score
    return max_score

def cum_log_ret_regime(regime, ret, n_classes=2, perm=None):
    """
    Compute cumulative log return based on a regime mapping.

    Parameters:
      regime: A sequence (list or array) where each element represents a regime (e.g., 0, 1, ...).
      ret: A sequence (numpy array or list) of returns that corresponds one-to-one with the regime.
      n_classes: The total number of classes (default is 2).
      perm: If not None, use the provided permutation to compute the score;
            if None, iterate over all possible permutations to find the best one.

    Returns:
      If perm is None, returns a tuple (max_score, best_perm) where:
         - max_score is the highest score obtained,
         - best_perm is the permutation (tuple) achieving that score.
      If perm is provided, returns the score computed using that permutation.
    """
    if perm is None:
        max_score = float('-inf')
        best_perm = None
        # Iterate over all possible permutations of range(n_classes)
        for candidate in permutations(range(n_classes)):
            # Map each element in regime using the current candidate permutation
            regime_perm = np.array([candidate[r] for r in regime])
            # Compute the cumulative log return
            score = np.sum(regime_perm * np.log(1 + ret))
            if score > max_score:
                max_score = score
                best_perm = candidate
        return max_score, best_perm
    else:
        # Use the provided permutation to map the regime
        regime_perm = np.array([perm[r] for r in regime])
        score = np.sum(regime_perm * np.log(1 + ret))
        return score
    
def calculate_ret(states, returns, n_classes=2, model_name='', daily=False):
    results = {}
    ret, perm_train = cum_log_ret_regime(states['train'].reshape(-1), returns['train'], n_classes=n_classes)
    if daily:
        ret = ret / len(states['train'])
        print(f"{model_name} Daily log return [train]: {100*ret:.4f}%")
    else:
        print(f"{model_name} Cum log return [train]: {ret:.4f}")
    results['train'] = ret
    for label in ['val', 'test']:
        if label in states and label in returns:
            ret = cum_log_ret_regime(states[label].reshape(-1), returns[label], n_classes=n_classes, perm=perm_train)
            if daily:
                ret = ret / len(states[label])
                print(f"{model_name} Daily log return [{label}]: {100*ret:.4f}%")
            else:
                print(f"{model_name} Cum log return [{label}]: {ret:.4f}")
            results[label] = ret
    return results
    
def calculate_forecast_ret(states, returns, n_classes=2, model_name='', daily=False):
    def regime_forecast(data):
        forecast = data[:, :-1].copy()
        zeros_col = np.zeros((forecast.shape[0], 1)).astype(int)
        forecast = np.hstack([zeros_col, forecast])
        return forecast
    states_forecast = {}
    for label in ['train', 'val', 'test']:
        if label in states and label in returns:
            states_forecast[label] = regime_forecast(states[label])
    return calculate_ret(states_forecast, returns, n_classes, model_name+' (forecast)', daily)
    
def calculate_turnover(states):
    """
    Calculate the turnover in each row of the given data.

    Parameters:
    regime (numpy array): A 2D numpy array where each row represents a sequence of values. For 2 states it is 0/1. For 3 states it is -1/0/1

    Returns:
    None (prints the jump counts)
    """
    for label in ['train', 'val', 'test']:
        if label in states:
            
            regime = states[label]
            # Compute the difference between adjacent elements in each row.
            # A nonzero difference indicates a transition.
            diff_regime = np.diff(regime, axis=1)

            # Count the number of transitions in each row (where diff_data is not zero).
            changes_per_row = np.sum(np.abs(diff_regime), axis=1)

            # Sum up the transitions across all rows.
            total_changes = np.sum(changes_per_row)
            daily_changes = total_changes / regime.shape[0]

            #print("Number of transitions in each row:", changes_per_row)
            print(f"Total number of transitions across all rows [{label}]:", total_changes)
            print(f"Average number of transitions per day: [{label}]", daily_changes)