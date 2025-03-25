import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from .mixture_vae import MixtureVAE

class VAEModule:
    """
    Encapsulates VAE's fit() and inference() methods.
    For demonstration purposes, the argmax of mu is used as the "predicted state."
    """
    def __init__(self, model_params):
        #self.model = SimpleVAE(input_dim, latent_dim)
        if model_params.name == 'mixture_vae':
            self.model = MixtureVAE(model_params)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)
        
  
    def fit(self, train_loader, lr=1e-3, epochs=5):
        optimizer = optim.Adam(self.model.parameters(), lr=lr)
        self.model.train()

        for epoch in range(epochs):
            total_loss = 0.0
            for x_batch, _ in train_loader:
                x_batch = x_batch.to(self.device)
                optimizer.zero_grad()
                x_hat = self.model(x_batch)
                loss = self.model.loss
                loss.backward()
                optimizer.step()
                total_loss += loss.item()
            avg_loss = total_loss / len(train_loader.dataset)
            if (epoch+1) % 20 == 0:
                print(f"Epoch [{epoch+1}/{epochs}], VAE Loss: {avg_loss:.4f}")

                
    def inference(self, data_loader):
        """
        Uses the argmax of mu as the state prediction; compares with the majority state
        of the true values in each window (for demonstration purposes only).
        """
        self.model.eval()
        all_pred_s = []
        all_true_s = []

        with torch.no_grad():
            for x_batch, s_batch in data_loader:
                x_batch = x_batch.to(self.device)
                s_batch = s_batch.to(self.device)

                pred_prob = self.model.get_s_prob(x_batch)
                pred_state = torch.argmax(pred_prob, dim=-1)

                all_pred_s.append(pred_state.cpu().numpy())
                all_true_s.append(s_batch.cpu().numpy())

        all_pred_s = np.concatenate(all_pred_s)
        all_true_s = np.concatenate(all_true_s)
        return all_true_s, all_pred_s
    
    
    def get_embedding(self, data_loader):
        """
        Uses the argmax of mu as the state prediction; compares with the majority state
        of the true values in each window (for demonstration purposes only).
        """
        self.model.eval()
        all_z = []
        all_x = []

        with torch.no_grad():
            for x_batch, _ in data_loader:
                x_batch = x_batch.to(self.device)
                all_x.append(x_batch.cpu().numpy())

                emd = self.model.get_z(x_batch)   # mu_z, logsigma2_z
                all_z.append(emd[0].cpu().numpy())

        all_x = np.concatenate(all_x)
        all_z = np.concatenate(all_z)
        return all_x, all_z