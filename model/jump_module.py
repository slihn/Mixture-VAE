import numpy as np
from .jump_model import JumpModel as JM
from .jump_model_batch import JumpModelBatch as JMB

class JumpModule:
    """
    A wrapper module using JumpModelBatch (or JumpModel) to fit and infer hidden states.
    """

    def __init__(self, 
                 n_states=2, 
                 jump_penalty=1e-5, 
                 max_iter=10, 
                 n_init=10, 
                 tol=None, 
                 verbose=False,
                 use_batch=True):
        """
        use_batch=True indicates using JumpModelBatch; otherwise, JumpModel is used.
        """
        self.use_batch = use_batch
        if self.use_batch:
            self.model = JMB(n_states=n_states,
                             jump_penalty=jump_penalty,
                             max_iter=max_iter,
                             n_init=n_init,
                             tol=tol,
                             verbose=verbose)
        else:
            self.model = JM(n_states=n_states,
                            jump_penalty=jump_penalty,
                            max_iter=max_iter,
                            n_init=n_init,
                            tol=tol,
                            verbose=verbose)
        
    def fit(self, train_loader):
        """
        Collects all (x_batch, _) from train_loader and calls fit of JumpModelBatch (or JumpModel).
        
        For JumpModelBatch:
          - x_batch.shape: (batch_size, window_size, D)
          - All batches are concatenated using np.concatenate(...) to get shape = (B_total, T, D)
          
        For JumpModel (single sequence version):
          - x_batch.shape: (batch_size, D), usually multiple slices of a single sequence
          - After concatenation, becomes (N, D) before fitting.
        """
        X_all = []
        
        for x_batch, _ in train_loader:
            # x_batch: (batch_size, window_size, D)
            # If tensor, convert to numpy first
            X_all.append(x_batch.numpy())
        
        # Concatenate along the batch dimension
        # JumpModelBatch requires shape (B, T, D)
        X_all = np.concatenate(X_all, axis=0)  # (B_total, T, D)
        
        # Call the model's fit
        self.model.fit(X_all)

    def inference(self, test_loader):
        """
        Performs model inference for each batch in test_loader
        and collects all true and predicted labels in shape (B, T).

        Returns
        -------
        all_true_s: np.ndarray, shape (B_total, T)
        all_pred_s: np.ndarray, shape (B_total, T)
        """
        all_true_s = []
        all_pred_s = []
        
        for x_batch, s_batch in test_loader:
            # x_batch: (B, T, D)
            # s_batch: (B, T)
            x_np = x_batch.numpy()
            s_np = s_batch.numpy()

            # JumpModelBatch's inference can handle the entire batch -> (B, T)
            pred_np = self.model.inference(x_np)  # shape: (B, T)
            
            # Directly collect data in shape (B, T)
            all_true_s.append(s_np)
            all_pred_s.append(pred_np)
        
        # Concatenate along the batch dimension
        # Final shape: (B_total, T)
        all_true_s = np.concatenate(all_true_s)
        all_pred_s = np.concatenate(all_pred_s)
        
        return all_true_s, all_pred_s
