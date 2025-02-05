import torch
import torch.nn as nn
import torch.nn.functional as F
from .blocks import *

class PrefixedMedNeXtBlock(nn.Module):

    def __init__(self, 
                in_channels:int, 
                out_channels:int, 
                exp_r:int=4, 
                kernel_size:int=7, 
                do_res:int=True,
                norm_type:str = 'group',
                n_groups:int or None = None,
                dim = '3d',
                grn = False,
                prefix: str='med_adp' # added prefix argument
                ):

        super().__init__()

        self.do_res = do_res

        self.prefix = prefix # added

        assert dim in ['2d', '3d']
        self.dim = dim
        if self.dim == '2d':
            conv = nn.Conv2d
        elif self.dim == '3d':
            conv = nn.Conv3d
            
        # First convolution layer with DepthWise Convolutions
        self.conv1 = conv(
            in_channels = in_channels,
            out_channels = in_channels,
            kernel_size = kernel_size,
            stride = 1,
            padding = kernel_size//2,
            groups = in_channels if n_groups is None else n_groups,
        )

        # Normalization Layer. GroupNorm is used by default.
        if norm_type=='group':
            self.norm = nn.GroupNorm(
                num_groups=in_channels, 
                num_channels=in_channels
                )
        elif norm_type=='layer':
            self.norm = LayerNorm(
                normalized_shape=in_channels, 
                data_format='channels_first'
                )

        # Second convolution (Expansion) layer with Conv3D 1x1x1
        self.conv2 = conv(
            in_channels = in_channels,
            out_channels = exp_r*in_channels,
            kernel_size = 1,
            stride = 1,
            padding = 0
        )
        
        # GeLU activations
        self.act = nn.GELU()
        
        # Third convolution (Compression) layer with Conv3D 1x1x1
        self.conv3 = conv(
            in_channels = exp_r*in_channels,
            out_channels = out_channels,
            kernel_size = 1,
            stride = 1,
            padding = 0
        )

        self.grn = grn
        if grn:
            if dim == '3d':
                self.grn_beta = nn.Parameter(torch.zeros(1,exp_r*in_channels,1,1,1), requires_grad=True)
                self.grn_gamma = nn.Parameter(torch.zeros(1,exp_r*in_channels,1,1,1), requires_grad=True)
            elif dim == '2d':
                self.grn_beta = nn.Parameter(torch.zeros(1,exp_r*in_channels,1,1), requires_grad=True)
                self.grn_gamma = nn.Parameter(torch.zeros(1,exp_r*in_channels,1,1), requires_grad=True)

        self._rename_layers() # added: call method to rename layers

    def _rename_layers(self): # added this method to rename layers
        """
        Rename layers by adding prefix to their names and updating module attributes.
        """
        # collect renaming operations
        rename_operations = []

        for name, module in self.named_children():
            new_name = f"{self.prefix}_{name}" # Add prefix to the layer name
            # print(f"name: {name}, new_name: {new_name}, module: {module}") # name: conv1, new_name: med_adp_conv1, module: Conv3d(4, 4, kernel_size=(3, 3, 3), stride=(1, 1, 1), padding=(1, 1, 1), groups=4)
            rename_operations.append((name, new_name, module))
        
        for name, new_name, module in rename_operations:
            setattr(self, new_name, module) # set the new name
            delattr(self, name) # Delete the old name


    def forward(self, x, dummy_tensor=None):
        
        x1 = x
        x1 = self.med_adp_conv1(x1) # changed_here from self.conv1(x1) to self.med_adp_conv1(x1)
        x1 = self.med_adp_act(self.med_adp_conv2(self.med_adp_norm(x1))) # changed
        if self.grn:
            # gamma, beta: learnable affine transform parameters
            # X: input of shape (N,C,H,W,D)
            if self.dim == '3d':
                gx = torch.norm(x1, p=2, dim=(-3, -2, -1), keepdim=True)
            elif self.dim == '2d':
                gx = torch.norm(x1, p=2, dim=(-2, -1), keepdim=True)
            nx = gx / (gx.mean(dim=1, keepdim=True)+1e-6)
            x1 = self.med_adp_grn_gamma * (x1 * nx) + self.med_adp_grn_beta + x1 # changed
        x1 = self.med_adp_conv3(x1) # changed
        if self.do_res:
            x1 = x + x1  
        return x1


if __name__ == "__main__":
    model = PrefixedMedNeXtBlock(
        in_channels=4,
        out_channels=4,
        exp_r=2,
        kernel_size=3,
        do_res=True,
        norm_type= 'group',
        dim='3d',
        grn=False,
        prefix='med_adp'
    )
    # Need to import block instead of .block when running this file individually
    # Parameters like self.do_res, self.dim, and self.grn are not layers or modules, so they don't need to be renamed