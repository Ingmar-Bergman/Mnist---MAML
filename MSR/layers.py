import math
import numpy as np
import torch
from torch import nn
from torch.nn import Module, init
from torch.nn.parameter import Parameter
import torch.nn.functional as F



class ShareConv2d(nn.Conv2d):
    def __init__(self, in_channels, out_channels, kernel_size, sigmoid=False, **kwargs):
        # Parameters:
        # in_channels: Number of input channels.
        # out_channels: Number of output channels.
        # kernel_size: Size of the convolutional kernel.
        # sigmoid: A boolean flag indicating whether to apply a sigmoid function to the warp matrices.
        # kwargs: Additional keyword arguments passed to the base nn.Conv2d class


        self.sigmoid = sigmoid
        self.A_warp, self.B_warp, self.C_warp = None, None, None
        super(ShareConv2d, self).__init__(in_channels, out_channels, kernel_size, **kwargs)
        
        # Calculate the size of the kernel
        k_size = int(np.prod(self.kernel_size))
        
        # Initialize warp matrices as identity matrices. Ce sont les matrices U.
        self.A_warp = nn.Parameter(torch.eye(self.out_channels, self.out_channels)) #transforme les canaux de sortie des poids de la convolution.
        self.B_warp = nn.Parameter(torch.eye(self.in_channels, self.in_channels)) #transforme les canaux d'entrée des poids de la convolution.
        self.C_warp = nn.Parameter(torch.eye(k_size, k_size)) #transforme les poids (les éléments du noyeau) de la convolution.
        
        # If sigmoid is True, initialize temp_warp
        if self.sigmoid:
            self.temp_warp = nn.Parameter(torch.rand(1))
        
        # Reset parameters
        self.reset_parameters()

    def reset_parameters(self):

        # Reset the parameters of the base Conv2d class
        super(ShareConv2d, self).reset_parameters()

        # Reset warp parameters
        if self.A_warp is not None:
            self.reset_warp_parameters()

    def reset_warp_parameters(self):

        # Initialize warp matrices with zeros
        init._no_grad_fill_(self.A_warp, 0.0)
        init._no_grad_fill_(self.B_warp, 0.0)
        init._no_grad_fill_(self.C_warp, 0.0)

        # If sigmoid is True, set specific initial values
        if self.sigmoid:
            init._no_grad_fill_(self.temp_warp, 7.0)
            init._no_grad_fill_(self.A_warp, -1.0)
            init._no_grad_fill_(self.B_warp, -1.0)
            init._no_grad_fill_(self.C_warp, -1.0)

        # Set the diagonal elements of warp matrices to 1 (identity matrices)
        with torch.no_grad():
            self.A_warp.fill_diagonal_(1.0)
            self.B_warp.fill_diagonal_(1.0)
            self.C_warp.fill_diagonal_(1.0)

    def forward(self, x):

        # Apply sigmoid to warp matrices if sigmoid flag is set
        
        A_warp, B_warp, C_warp = self.A_warp, self.B_warp, self.C_warp
        if self.sigmoid:
            A_warp = torch.sigmoid(self.temp_warp * A_warp)
            B_warp = torch.sigmoid(self.temp_warp * B_warp)
            C_warp = torch.sigmoid(self.temp_warp * C_warp)

        # Reshape weights to (out_channels, in_channels, kernel_size * kernel_size)
        orig_shape = self.weight.shape
        # (c_o, c_i, k_w * k_h)
        weight = torch.reshape(self.weight, (self.out_channels, self.in_channels, -1))

        # Apply warp matrices to the weights using Einstein summation
        weight = torch.einsum("ij,jkl->ikl", A_warp, weight)
        weight = torch.einsum("ik,jkl->jil", B_warp, weight)
        weight = torch.einsum("il,jkl->jki", C_warp, weight)

        # Reshape weights back to the original shape
        weight = torch.reshape(weight, orig_shape)

        # Perform the convolution operation with the warped weights
        return self._conv_forward(x, weight, bias=self.bias) # return self.conv2d_forward(x, weight)marche plus depuis la version 1.8.0 de pytorch
        # return self._conv_forward(x, weight, bias=self.bias)  #MH notebook