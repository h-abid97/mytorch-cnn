import numpy as np
from resampling import *


class Conv2d_stride1():
    def __init__(self, in_channels, out_channels,
                 kernel_size, weight_init_fn=None, bias_init_fn=None):

        # Do not modify this method

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size

        if weight_init_fn is None:
            self.W = np.random.normal(
                0, 1.0, (out_channels, in_channels, kernel_size, kernel_size))
        else:
            self.W = weight_init_fn(
                out_channels,
                in_channels,
                kernel_size,
                kernel_size)

        if bias_init_fn is None:
            self.b = np.zeros(out_channels)
        else:
            self.b = bias_init_fn(out_channels)

        self.dLdW = np.zeros(self.W.shape)
        self.dLdb = np.zeros(self.b.shape)

    def forward(self, A):
        """
        Argument:
            A (np.array): (batch_size, in_channels, input_height, input_width)
        Return:
            Z (np.array): (batch_size, out_channels, output_height, output_width)
        """
        self.A = A
        batch_size, _, input_height, input_width = A.shape
        
        output_height = input_height - self.kernel_size + 1
        output_width = input_width - self.kernel_size + 1

        # Initialize the output tensor Z with zeros.
        Z = np.zeros((batch_size, self.out_channels, output_height, output_width))

        # Iterate over each position in the output tensor to apply the convolution operation.
        for i in range(output_height):
            for j in range(output_width):
                # Perform the tensor dot product between the input and the kernel weights (self.W) for each position.
                # This operation computes the convolution for a subregion of the input tensor.
                Z[:, :, i, j] = np.tensordot(A[:, :, i:(i + self.kernel_size), j:(j + self.kernel_size)], self.W, axes=[(1, 2, 3), (1, 2, 3)])

        # Prepare the bias tensor to be the same shape as the output height and width. This allows for the addition
        # of the bias to the convolution result. The bias is broadcasted across the batch and output channels.
        b = np.vstack([self.b] * output_height).T
        b = np.dstack([b] * output_width)
        
        # Add the bias to the convolution result to get the final output tensor Z.
        Z += b

        return Z

    def backward(self, dLdZ):
        """
        Argument:
            dLdZ (np.array): (batch_size, out_channels, output_height, output_width)
        Return:
            dLdA (np.array): (batch_size, in_channels, input_height, input_width)
        """
        batch_size, _, output_height, output_width = dLdZ.shape 
        _, _, input_height, input_width = self.A.shape

        self.dLdW = np.zeros(self.W.shape)
        self.dLdb = np.zeros(self.b.shape)
        dLdA = np.zeros((batch_size, self.in_channels, input_height, input_width))

        # Calculate the necessary padding based on the kernel size to ensure the dimensions match for the convolution.
        zero_padding = (self.kernel_size - 1) * 2
        dLdZ_padded = np.pad(dLdZ, ((0, 0), (0, 0), (zero_padding // 2, zero_padding // 2), (zero_padding // 2, zero_padding // 2)), 'constant')

        # Flip the weights for the convolution operation in the backward pass.
        flipped_W = np.flip(self.W, axis=(2, 3))

        # Calculate the gradient of the loss with respect to the input tensor A (dLdA) 
        # by convolving dLdZ_padded with flipped weights.
        for i in range(input_height):
            for j in range(input_width):
                dLdA[:, :, i, j] = np.tensordot(dLdZ_padded[:, :, i:(i + self.kernel_size), j:(j + self.kernel_size)], 
                                                flipped_W, axes=[(1, 2, 3), (0, 2, 3)])
        
        # Compute the gradient with respect to the weights (dLdW) by convolving the input tensor A 
        # with the gradient of the loss w.r.t. the output tensor.
        for i in range(self.dLdW.shape[2]):
            for j in range(self.dLdW.shape[3]):
                self.dLdW[:, :, i, j] = np.tensordot(self.A[:, :, i:(i + output_height), j:(j + output_width)], 
                                                    dLdZ, axes=[(0, 2, 3), (0, 2, 3)]).T
        
        # Calculate the gradient of the loss with respect to the biases (dLdb) by summing dLdZ 
        # over the batch, height, and width dimensions.
        self.dLdb = np.sum(dLdZ, axis=(0, 2, 3))

        return dLdA


class Conv2d():
    def __init__(self, in_channels, out_channels, kernel_size, stride,
                 weight_init_fn=None, bias_init_fn=None):
        # Do not modify the variable names
        self.stride = stride

        # Initialize Conv2d() and Downsample2d() isntance
        self.conv2d_stride1 = Conv2d_stride1(in_channels, out_channels, kernel_size, weight_init_fn, bias_init_fn)
        self.downsample2d = Downsample2d(stride)

    def forward(self, A):
        """
        Argument:
            A (np.array): (batch_size, in_channels, input_height, input_width)
        Return:
            Z (np.array): (batch_size, out_channels, output_height, output_width)
        """
        # Call Conv2d_stride1
        Z = self.conv2d_stride1.forward(A)

        # downsample
        Z = self.downsample2d.forward(Z)

        return Z

    def backward(self, dLdZ):
        """
        Argument:
            dLdZ (np.array): (batch_size, out_channels, output_height, output_width)
        Return:
            dLdA (np.array): (batch_size, in_channels, input_height, input_width)
        """
        # Call downsample1d backward
        dLdA_downsampled = self.downsample2d.backward(dLdZ)

        # Call Conv1d_stride1 backward
        dLdA = self.conv2d_stride1.backward(dLdA_downsampled)

        return dLdA
