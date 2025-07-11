# Do not import any additional 3rd party external libraries as they will not
# be available to AutoLab and are not needed (or allowed)

import numpy as np
from resampling import *


class Conv1d_stride1():
    def __init__(self, in_channels, out_channels, kernel_size,
                 weight_init_fn=None, bias_init_fn=None):
        # Do not modify this method
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size

        if weight_init_fn is None:
            self.W = np.random.normal(
                0, 1.0, (out_channels, in_channels, kernel_size))
        else:
            self.W = weight_init_fn(out_channels, in_channels, kernel_size)

        if bias_init_fn is None:
            self.b = np.zeros(out_channels)
        else:
            self.b = bias_init_fn(out_channels)

        self.dLdW = np.zeros(self.W.shape)
        self.dLdb = np.zeros(self.b.shape)

    def forward(self, A):
        """
        Argument:
            A (np.array): (batch_size, in_channels, input_size)
        Return:
            Z (np.array): (batch_size, out_channels, output_size)
        """
        self.A = A

        batch_size, _, input_size = self.A.shape
        out_channels, _, kernel_size = self.W.shape

        output_size = input_size - kernel_size + 1
        Z = np.zeros((batch_size, out_channels, output_size))

        for i in range(output_size):
            Z[:, :, i] = np.tensordot(A[:, :, i:(i + kernel_size)], self.W, axes=[(1, 2), (1, 2)])
        
        b = np.vstack([self.b] * output_size).T
        Z += b

        return Z

    def backward(self, dLdZ):
        """
        Argument:
            dLdZ (np.array): (batch_size, out_channels, output_size)
        Return:
            dLdA (np.array): (batch_size, in_channels, input_size)
        """
        batch_size, _, output_size = dLdZ.shape
        input_size = self.A.shape[2]

        self.dLdW = np.zeros(self.W.shape)
        self.dLdb = np.zeros(self.b.shape)
        dLdA = np.zeros((batch_size, self.in_channels, input_size))
       
        zero_padding = (self.kernel_size - 1) * 2
        dLdZ_padded = np.pad(dLdZ, ((0, 0), (0, 0), (zero_padding // 2, zero_padding // 2)), 'constant')

        flipped_W = np.flip(self.W, axis=2)

        for i in range(input_size):
            dLdA[:, :, i] = np.tensordot(dLdZ_padded[:, :, i:(i + self.kernel_size)], flipped_W, axes=[(1, 2), (0, 2)])
        
        for i in range(self.dLdW.shape[2]):
            self.dLdW[:, :, i] = np.tensordot(self.A[:, :, i:(i + output_size)], dLdZ, axes=[(0, 2), (0, 2)]).T
        
        self.dLdb = np.sum(dLdZ, axis=(0, 2))

        return dLdA


class Conv1d():
    def __init__(self, in_channels, out_channels, kernel_size, stride,
                 weight_init_fn=None, bias_init_fn=None):
        # Do not modify the variable names

        self.stride = stride

        # Initialize Conv1d() and Downsample1d() isntance
        self.conv1d_stride1 = Conv1d_stride1(in_channels, out_channels, kernel_size, weight_init_fn, bias_init_fn)
        self.downsample1d = Downsample1d(stride)

    def forward(self, A):
        """
        Argument:
            A (np.array): (batch_size, in_channels, input_size)
        Return:
            Z (np.array): (batch_size, out_channels, output_size)
        """

        # Call Conv1d_stride1
        Z = self.conv1d_stride1.forward(A)

        # downsample
        Z = self.downsample1d.forward(Z)

        return Z

    def backward(self, dLdZ):
        """
        Argument:
            dLdZ (np.array): (batch_size, out_channels, output_size)
        Return:
            dLdA (np.array): (batch_size, in_channels, input_size)
        """
        # Call downsample1d backward
        dLdA_downsampled = self.downsample1d.backward(dLdZ)

        # Call Conv1d_stride1 backward
        dLdA = self.conv1d_stride1.backward(dLdA_downsampled)

        return dLdA
