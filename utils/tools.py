import torch

def count_parameters(model):
    """Counts the total and trainable parameters of a PyTorch model.

    Args:
        model: A PyTorch model (e.g., an instance of nn.Module).

    Returns:
        A tuple containing:
        - total_params: The total number of parameters in the model.
        - trainable_params: The number of trainable parameters in the model.
    """
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total number of parameters: {total_params}")
    print(f"Number of trainable parameters: {trainable_params}")
    return total_params, trainable_params