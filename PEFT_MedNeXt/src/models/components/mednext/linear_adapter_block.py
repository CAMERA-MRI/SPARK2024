import torch
import torch.nn as nn
import torch.nn.functional as F


class LinearAdapter(nn.Module):
    """
    Args:
        input_dim (int): Dimension of input features
        adapter_dim (int): Intermediate dimension for adapter
        use_gelu (bool): Flag for using gelu activation or not. Default is False
    """

    def __init__(self, input_dim: int, adapter_dim: int, use_gelu=False):
        super().__init__()

        self.input_dim = input_dim
        self.adapter_dim = adapter_dim

        self.fc1 = nn.Linear(self.input_dim, self.adapter_dim)
        self.fc2 = nn.Linear(self.adapter_dim, self.input_dim)
        self.activation = nn.GELU() if use_gelu else nn.ReLU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Pretrained Features
        Returns: Adapted Features for the task
        """
        orig_shape = x.shape # orig_shape =>[B, C, H, W, D] for instance, torch.Size([1, 32, 128, 128, 128])

        assert self.input_dim == orig_shape[1]
        
        # Flatten the spatial_dimension such that x -> [B*H*W*D, C], if x is 5D tensors i.e. x -> [B, C, H, W, D]
        if len(orig_shape) == 5:
            x_flat = x.view(-1, self.input_dim) # for instance, x_flat of shape torch.Size([2097152, 32]) 
        else:
            raise ValueError(f"Unexpected input shape: {orig_shape}")
        
        h = self.activation(self.fc1(x_flat)) # [2097152, 32].[32, adapter_dim=8] => [2097152, adapter_dim=8]
        h = self.activation(self.fc2(h)) # [2097152, 8].[8, 32] => [2097152, 32]

        # Reshape back to original dimension
        if len(orig_shape) == 5:
            h = h.view(*orig_shape) # [1, 32, 128, 128, 128]

        return x + h