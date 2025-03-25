import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils
import torch.distributions
import torchvision
from torch import Tensor
from typing import List, Tuple, Any
import numpy as np

# =============================================================================
# Basic NN Module
# =============================================================================

# Multi-Layer Perceptron (MLP) Module
class MLP(nn.Module):
    """
    Parameters:
        in_dim (int): Input feature dimension.
        hiddens (List[int]): List of hidden layer dimensions, e.g. [32, 128, 128, 64].
        out_dim (int): Output dimension.
        dropout (float, optional): Dropout probability (default is 0.0, i.e., no dropout).
        activation (nn.Module, optional): Activation function (default is nn.ReLU()).
    """
    def __init__(self, in_dim: int, hiddens: List[int], out_dim: int,
                 dropout: float = 0.0, activation: nn.Module = nn.ReLU()):
        super(MLP, self).__init__()
        layers = []
        for hidden_dim in hiddens:
            layers.append(nn.Linear(in_dim, hidden_dim))
            layers.append(activation)
            layers.append(nn.Dropout(dropout))
            in_dim = hidden_dim
        # Final linear layer mapping to the output dimension
        layers.append(nn.Linear(in_dim, out_dim))
        self.model = nn.Sequential(*layers)

    def forward(self, x: Tensor) -> Tensor:
        """
        Forward pass through the MLP.
        
        Args:
            x (Tensor): Input tensor.
            
        Returns:
            Tensor: Output tensor.
        """
        return self.model(x)

    
# LSTM Module (Causal)
class LSTMNet(nn.Module):
    """
    An LSTM-based module for sequence modeling. It is configured to be causal,
    meaning it only uses current and previous time steps.
    
    Parameters:
        in_dim (int): Input feature dimension.
        lstm_hidden (int): Hidden state dimension of the LSTM.
        lstm_layers (int): Number of LSTM layers.
        out_dim (int): Output dimension.
        dropout (float, optional): Dropout probability (applied both internally in LSTM and after LSTM).
    """
    def __init__(self, in_dim: int, lstm_hidden: int, lstm_layers: int,
                 out_dim: int, dropout: float = 0.0):
        super(LSTMNet, self).__init__()
        self.lstm = nn.LSTM(
            input_size=in_dim,
            hidden_size=lstm_hidden,
            num_layers=lstm_layers,
            dropout=dropout if lstm_layers > 1 else 0.0,  # Dropout is only effective if num_layers > 1.
            batch_first=True,
            bidirectional=False  # Single direction to ensure causality.
        )
        self.dropout = nn.Dropout(dropout)
        self.linear = nn.Linear(lstm_hidden, out_dim)

    def forward(self, x: Tensor) -> Tensor:
        """
        Forward pass through the LSTM network.
        
        Args:
            x (Tensor): Input tensor of shape [batch, seq_len, in_dim].
            
        Returns:
            Tensor: Output tensor of shape [batch, seq_len, out_dim].
        """
        lstm_out, _ = self.lstm(x)  # lstm_out: [batch, seq_len, lstm_hidden]
        lstm_out = self.dropout(lstm_out)
        out = self.linear(lstm_out)  # out: [batch, seq_len, out_dim]
        return out

# =============================================================================
# BaseNet: Module Selector (MLP or LSTMNet)
# =============================================================================
class BaseNet(nn.Module):
    """
    Base network class that selects the appropriate module (MLP or LSTMNet) based on provided arguments.
    
    Parameters:
        args (Any): Configuration object containing attributes.
        in_dim (int): Input feature dimension.
        out_dim (int): Output dimension.
        module (str): Module identifier used to fetch configuration parameters.
    """
    def __init__(self, args: Any, in_dim: int, out_dim: int, module: str):
        super(BaseNet, self).__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim

        net_type = getattr(args, f"{module}_type").lower()
        if net_type == "lstm":
            lstm_hidden = getattr(args, f"{module}_lstm_hidden")
            lstm_layers = getattr(args, f"{module}_lstm_layers")
            lstm_dropout = getattr(args, f"{module}_dropout")
            self.net = LSTMNet(
                in_dim=self.in_dim,
                lstm_hidden=lstm_hidden,
                lstm_layers=lstm_layers,
                out_dim=self.out_dim,
                dropout=lstm_dropout
            )
        else:
            hiddens = getattr(args, f"{module}_hiddens")
            dropout = getattr(args, f"{module}_dropout")
            self.net = MLP(
                in_dim=self.in_dim,
                hiddens=hiddens,
                out_dim=self.out_dim,
                dropout=dropout,
                activation=nn.ReLU()
            )

    def forward(self, x: Tensor) -> Tensor:
        """
        Forward pass through the selected network.
        
        Args:
            x (Tensor): Input tensor.
            
        Returns:
            Tensor: Output tensor.
        """
        return self.net(x)

# =============================================================================
# Basic Modulesb for VAE
# =============================================================================

# S_X: Soft Cluster Assignment Module
class S_X(nn.Module):
    """
    Module for computing soft cluster assignment probabilities.
    
    Parameters:
        args (Any): Configuration object containing attributes including:
            - s_clamp: Value to clamp the logits.
            - feature: Input feature dimension.
            - n_cluster: Number of clusters.
            - s_x_type, s_x_hiddens, s_x_dropout, etc. for network configuration.
    """
    def __init__(self, args: Any):
        super(S_X, self).__init__()
        self.s_clamp = args.s_clamp
        self.net = BaseNet(args, in_dim=args.feature, out_dim=args.n_cluster, module='s_x')

    def forward(self, x: Tensor) -> Tensor:
        """
        Forward pass that computes the cluster assignment probabilities.
        
        Args:
            x (Tensor): Input tensor.
            
        Returns:
            Tensor: Softmax probabilities over clusters.
        """
        x = self.net(x)
        x = torch.clamp(x, min=-self.s_clamp, max=self.s_clamp)
        x = F.softmax(x, dim=-1)
        return x

    
# Z_S: Latent Parameter Estimation from Clusters
class Z_S(nn.Module):
    """
    Module to compute latent variable parameters from cluster assignments.
    
    Parameters:
        args (Any): Configuration object containing attributes:
            - n_cluster: Number of clusters.
            - hidden_dim: Dimension of the latent space.
    """
    def __init__(self, args: Any):
        super(Z_S, self).__init__()
        self.n_cluster = args.n_cluster
        self.hidden_dim = args.hidden_dim
        self.mu = nn.Linear(self.n_cluster, self.hidden_dim)
        self.logsigma2 = nn.Linear(self.n_cluster, self.hidden_dim)

    def forward(self, s: Tensor) -> Tuple[Tensor, Tensor]:
        """
        Forward pass that computes the mean and log variance of the latent variable.
        
        Args:
            s (Tensor): Cluster assignments tensor of shape [batch, seq_len, n_cluster].
            
        Returns:
            Tuple[Tensor, Tensor]: Mean and log variance tensors, each of shape [batch, seq_len, hidden_dim].
        """
        return self.mu(s), self.logsigma2(s)


# Z_SX: Latent Parameter Estimation Conditioned on s and x
class Z_SX(nn.Module):
    """
    Module to compute latent variable parameters conditioned on both cluster assignments and input features.
    
    Parameters:
        args (Any): Configuration object containing attributes:
            - feature: Input feature dimension.
            - n_cluster: Number of clusters.
            - hidden_dim: Dimension of the latent space.
            - z_sx_type, z_sx_hiddens, z_sx_dropout, etc. for network configuration.
    """
    def __init__(self, args: Any):
        super(Z_SX, self).__init__()
        self.hidden_dim = args.hidden_dim
        self.net = BaseNet(args, in_dim=args.feature + args.n_cluster, out_dim=2 * args.hidden_dim, module='z_sx')

    def forward(self, s: Tensor, x: Tensor) -> Tuple[Tensor, Tensor]:
        """
        Forward pass that computes the mean and log variance of the latent variable conditioned on s and x.
        
        Args:
            s (Tensor): Cluster assignments tensor of shape [batch, seq_len, n_cluster].
            x (Tensor): Input features tensor of shape [batch, seq_len, feature].
            
        Returns:
            Tuple[Tensor, Tensor]: Mean and log variance tensors, each of shape [batch, seq_len, hidden_dim].
        """
        x_cat = torch.cat((s, x), dim=-1)     # Shape: [batch, seq_len, n_cluster + feature]
        x_out = self.net(x_cat)                # Shape: [batch, seq_len, 2 * hidden_dim]
        mu = x_out[:, :, :self.hidden_dim]
        logsigma2 = x_out[:, :, self.hidden_dim:]
        return mu, logsigma2


# X_SZ: Reconstruction Module Using s and z
class X_SZ(nn.Module):
    """
    Module to reconstruct input features from cluster assignments and latent variable parameters.
    
    Parameters:
        args (Any): Configuration object containing attributes:
            - hidden_dim: Dimension of the latent space.
            - n_cluster: Number of clusters.
            - feature: Input feature dimension.
            - x_sz_type, x_sz_hiddens, x_sz_dropout, etc. for network configuration.
    """
    def __init__(self, args: Any):
        super(X_SZ, self).__init__()
        self.net = BaseNet(args, in_dim=2 * args.hidden_dim + args.n_cluster, out_dim=args.feature, module='x_sz')

    def forward(self, s: Tensor, mu_z: Tensor, logsigma2_z: Tensor) -> Tensor:
        """
        Forward pass that reconstructs the input features.
        
        Args:
            s (Tensor): Cluster assignments tensor of shape [batch, seq_len, n_cluster].
            mu_z (Tensor): Latent mean tensor of shape [batch, seq_len, hidden_dim].
            logsigma2_z (Tensor): Latent log variance tensor of shape [batch, seq_len, hidden_dim].
            
        Returns:
            Tensor: Reconstructed features tensor of shape [batch, seq_len, feature].
        """
        x_cat = torch.cat((s, mu_z, logsigma2_z), dim=-1)  # Shape: [batch, seq_len, 2 * hidden_dim + n_cluster]
        out = self.net(x_cat)
        return out


# X_Z: Reconstruction Module Using Only z
class X_Z(nn.Module):
    """
    Module to reconstruct input features solely from latent variable parameters.
    
    Parameters:
        args (Any): Configuration object containing attributes:
            - hidden_dim: Dimension of the latent space.
            - feature: Input feature dimension.
            - x_z_type, x_z_hiddens, x_z_dropout, etc. for network configuration.
    """
    def __init__(self, args: Any):
        super(X_Z, self).__init__()
        self.net = BaseNet(args, in_dim=2 * args.hidden_dim, out_dim=args.feature, module='x_z')

    def forward(self, mu_z: Tensor, logsigma2_z: Tensor) -> Tensor:
        """
        Forward pass that reconstructs the input features.
        
        Args:
            mu_z (Tensor): Latent mean tensor of shape [batch, seq_len, hidden_dim].
            logsigma2_z (Tensor): Latent log variance tensor of shape [batch, seq_len, hidden_dim].
            
        Returns:
            Tensor: Reconstructed features tensor of shape [batch, seq_len, feature].
        """
        x_cat = torch.cat((mu_z, logsigma2_z), dim=-1)  # Shape: [batch, seq_len, 2 * hidden_dim]
        out = self.net(x_cat)
        return out

# =============================================================================
# MixtureVAE: Main Model Combining All Components
# =============================================================================
class MixtureVAE(nn.Module):
    """
    A Mixture Variational Autoencoder (VAE) that integrates clustering and latent variable modeling.
    
    This module includes:
      - s_x: Computes soft cluster assignments.
      - z_s: Computes latent embeddings from cluster assignments.
      - z_sx: Computes latent embeddings conditioned on both cluster assignments and input features.
      - x_sz: Reconstructs input features using cluster assignments and latent embeddings.
      - x_z: Reconstructs input features using latent embeddings only.
    
    The loss is composed of reconstruction loss, mixture loss, information loss, and transition loss.
    
    Parameters:
        args (Any): Configuration object containing necessary attributes.
    """
    def __init__(self, args: Any):
        super(MixtureVAE, self).__init__()
        self.args = args
        self.s_x = S_X(args)
        self.z_s = Z_S(args)
        self.z_sx = Z_SX(args)
        if self.args.reconstruction_on_s:
            self.x_sz = X_SZ(args)
        else:
            self.x_z = X_Z(args)
        self.loss = 0.0

    def forward(self, x: Tensor) -> Tensor:
        """
        Forward pass through the Mixture VAE.
        
        Args:
            x (Tensor): Input tensor of shape [batch, seq_len, feature].
            
        Returns:
            Tensor: Reconstructed features of shape [batch, seq_len, feature].
        """
        # Compute soft cluster assignment probabilities.
        s_prob = self.s_x(x)  # Shape: [batch, seq_len, n_cluster]
        
        # Sample cluster assignments using Gumbel-Softmax.
        s = F.gumbel_softmax(torch.log(s_prob), tau=self.args.tau, hard=self.args.hard)  # Shape: [batch, seq_len, n_cluster]
        
        # Compute latent variable parameters.
        mu_z_q, logsigma2_z_q = self.z_sx(s, x)  # Inference network (q): each tensor has shape [batch, seq_len, hidden_dim]
        mu_z_p, logsigma2_z_p = self.z_s(s)        # Prior network (p): each tensor has shape [batch, seq_len, hidden_dim]
        
        # Reconstruct the input based on the configuration.
        if self.args.reconstruction_on_s:
            if self.args.reconstruction_on_z == 'q':
                out = self.x_sz(s, mu_z_q, logsigma2_z_q)
            elif self.args.reconstruction_on_z == 'p':
                out = self.x_sz(s, mu_z_p, logsigma2_z_p)
            else:
                raise ValueError("Invalid value for reconstruction_on_z. Expected 'q' or 'p'.")
        else:
            if self.args.reconstruction_on_z == 'q':
                out = self.x_z(mu_z_q, logsigma2_z_q)
            elif self.args.reconstruction_on_z == 'p':
                out = self.x_z(mu_z_p, logsigma2_z_p)
            else:
                raise ValueError("Invalid value for reconstruction_on_z. Expected 'q' or 'p'.")
        
        # Compute losses.
        loss_i = information_loss(s_prob, self.args)  # Information loss.
        loss_t = transition_loss(s_prob, self.args)  # Transition loss.
        loss_m = mixture_loss(mu_z_q, logsigma2_z_q, mu_z_p, logsigma2_z_p, self.args)  # Mixture loss.
        loss_r = reconstruction_loss(out, x, self.args)  # Reconstruction loss.
        
        # Clamp losses.
        if self.args.loss_clamp != None:
            loss_i = torch.clamp(loss_i, min=-self.args.loss_clamp, max=self.args.loss_clamp)
            loss_t = torch.clamp(loss_t, min=-self.args.loss_clamp, max=self.args.loss_clamp)
            loss_m = torch.clamp(loss_m, min=-self.args.loss_clamp, max=self.args.loss_clamp)
        
        # Total loss as weighted sum of individual losses.
        self.loss = loss_r + self.args.lamda_m * loss_m + self.args.lamda_i * loss_i + self.args.lamda_t * loss_t
        return out

    def get_s_prob(self, x: Tensor) -> Tensor:
        """
        Retrieve the soft cluster assignment probabilities for the input.
        
        Args:
            x (Tensor): Input tensor of shape [batch, seq_len, feature].
            
        Returns:
            Tensor: Soft cluster probabilities of shape [batch, seq_len, n_cluster].
        """
        s_prob = self.s_x(x)
        return s_prob

    def get_z(self, x: Tensor) -> Tuple[Tensor, Tensor]:
        """
        Retrieve the latent variable parameters for the input based on sprob (not s).
        
        Note: This function is not fully implemented and may require additional logic based on configuration.
        
        Args:
            x (Tensor): Input tensor.
        """
        s_prob = self.s_x(x)
        mu_z_q, logsigma2_z_q = self.z_sx(s_prob, x)
        return mu_z_q, logsigma2_z_q

# =============================================================================
# Loss Functions
# =============================================================================
def information_loss(s_prob: Tensor, args) -> Tensor:
    """
    Compute the information loss given the soft cluster assignment probabilities.

    Args:
        s_prob (Tensor): Soft cluster probabilities.
        args: Object containing loss_mode.

    Returns:
        Tensor: Information loss.
    """
    loss = s_prob * torch.log(s_prob)
    if args.loss_mode == 'sum':
        return loss.sum()
    elif args.loss_mode == 'mean':
        return loss.mean()
    elif args.loss_mode == 'norm':
        return loss.sum()
    else:
        raise ValueError(f"Invalid loss_mode: {args.loss_mode}. Must be 'sum', 'mean', or 'norm'.")


def mixture_loss(mu_z_q: Tensor, logsigma2_z_q: Tensor, mu_z_p: Tensor, logsigma2_z_p: Tensor, args) -> Tensor:
    """
    Compute the mixture loss.

    Args:
        mu_z_q (Tensor): Mean of the encoder distribution.
        logsigma2_z_q (Tensor): Log variance of the encoder distribution.
        mu_z_p (Tensor): Mean of the prior distribution.
        logsigma2_z_p (Tensor): Log variance of the prior distribution.
        args: Object containing loss_mode.

    Returns:
        Tensor: Mixture loss.
    """
    ln2 = torch.log(torch.tensor(2.0))
    term0 = 0.5 * (logsigma2_z_p - logsigma2_z_q)
    term1 = torch.exp(logsigma2_z_q - logsigma2_z_p - 2 * ln2)
    term2 = (mu_z_q - mu_z_p) ** 2 / (2 * torch.exp(logsigma2_z_p))
    loss = term0 + term1 + term2

    if args.loss_mode == 'sum':
        return loss.sum()
    elif args.loss_mode == 'mean':
        return loss.mean()
    elif args.loss_mode == 'norm':
        return loss.sum() / np.sqrt(term0.size(-1))
    else:
        raise ValueError(f"Invalid loss_mode: {args.loss_mode}. Must be 'sum', 'mean', or 'norm'.")


def reconstruction_loss(out: Tensor, x: Tensor, args) -> Tensor:
    """
    Compute the reconstruction loss.

    Args:
        out (Tensor): Reconstructed input.
        x (Tensor): Original input.
        args: Object containing loss_mode.

    Returns:
        Tensor: Reconstruction loss.
    """
    loss = (out - x) ** 2
    if args.loss_mode == 'sum':
        return loss.sum()
    elif args.loss_mode == 'mean':
        return loss.mean()
    elif args.loss_mode == 'norm':
        return loss.sum() / np.sqrt(x.size(-1))
    else:
        raise ValueError(f"Invalid loss_mode: {args.loss_mode}. Must be 'sum', 'mean', or 'norm'.")


def transition_loss(s_prob: Tensor, args) -> Tensor:
    """
    Compute the transition loss.  Currently only supports the 'jump' transition.

    Args:
        s_prob (Tensor): Soft cluster probabilities.
        args: Object containing transition and loss_mode, and optionally jump_mx.

    Returns:
        Tensor: Transition loss.

    Raises:
        ValueError: If an invalid transition type or loss_mode is specified.
    """
    if args.transition == 'jump':
        if hasattr(args, 'jump_mx') and args.jump_mx is not None:
            jump_mx = torch.tensor(args.jump_mx, dtype=torch.float32, device=s_prob.device)
        else:
            n = s_prob.size(-1)
            jump_mx = torch.ones(n, n, device=s_prob.device) - torch.eye(n, device=s_prob.device)
            jump_mx = jump_mx.float()

        s0 = s_prob[:, :-1, :]
        s1 = s_prob[:, 1:, :]
        p = torch.matmul(s0.unsqueeze(-1), s1.unsqueeze(-2))
        p = (p * jump_mx).sum(dim=(-1, -2))

        if args.loss_mode == 'sum':
            return p.sum()
        elif args.loss_mode == 'mean':
            return p.mean()
        elif args.loss_mode == 'norm':
            return p.sum() / (s_prob.size(0) * (s_prob.size(1) - 1))
        else:
            raise ValueError(f"Invalid loss_mode: {args.loss_mode}. Must be 'sum', 'mean', or 'norm'.")
    else:
        raise ValueError(f"Invalid transition type: {args.transition}. Only 'jump' is currently supported.")

