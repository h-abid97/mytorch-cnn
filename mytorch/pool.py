import numpy as np
from resampling import *


class MaxPool2d_stride1():

    def __init__(self, kernel):
        self.kernel = kernel

    def forward(self, A):
        """
        Argument:
            A (np.array): (batch_size, in_channels, input_width, input_height)
        Return:
            Z (np.array): (batch_size, out_channels, output_width, output_height)
        """
        self.A_shape = A.shape
        batch_size, in_channels, input_width, input_height = A.shape

        output_width = input_width - self.kernel + 1
        output_height = input_height - self.kernel + 1
        
        Z = np.zeros((batch_size, in_channels, output_width, output_height))
        self.max_indices = np.zeros((batch_size, in_channels, output_width, output_height, 2), dtype=int)

        for i in range(output_width):
            for j in range(output_height):
                # Extract the window for max pooling
                patch = A[:, :, i:(i + self.kernel), j:(j + self.kernel)]
                
                # Perform max pooling
                Z[:, :, i, j] = np.max(patch, axis=(2, 3))
                
                # Find the indices of the max values
                max_positions = np.argmax(patch.reshape(batch_size, in_channels, -1), axis=2)
                
                # Convert the indices into two-dimensional indices
                index_y = max_positions // self.kernel
                index_x = max_positions % self.kernel
                
                # Store the indices
                self.max_indices[:, :, i, j, 0] = index_y
                self.max_indices[:, :, i, j, 1] = index_x 


        return Z

    def backward(self, dLdZ):
        """
        Argument:
            dLdZ (np.array): (batch_size, out_channels, output_width, output_height)
        Return:
            dLdA (np.array): (batch_size, in_channels, input_width, input_height)
        """
        batch_size, out_channels, output_width, output_height = dLdZ.shape

        dLdA = np.zeros(self.A_shape)

        for n in range(batch_size):
            for c in range(out_channels):
                for i in range(output_width):
                    for j in range(output_height):
                        # Retrieve the indices of the max value for the current window
                        (max_i, max_j) = self.max_indices[n, c, i, j]
                        # Only the max value receives gradient from dLdZ
                        dLdA[n, c, i + max_i, j + max_j] += dLdZ[n, c, i, j]

        return dLdA


class MeanPool2d_stride1():

    def __init__(self, kernel):
        self.kernel = kernel

    def forward(self, A):
        """
        Argument:
            A (np.array): (batch_size, in_channels, input_width, input_height)
        Return:
            Z (np.array): (batch_size, out_channels, output_width, output_height)
        """
        self.A_shape = A.shape
        batch_size, in_channels, input_width, input_height = A.shape

        output_width = input_width - self.kernel + 1
        output_height = input_height - self.kernel + 1
        
        Z = np.zeros((batch_size, in_channels, output_width, output_height))

        for i in range(output_width):
            for j in range(output_height):
                Z[:, :, i, j] = np.mean(A[:, :, i:(i + self.kernel), j:(j + self.kernel)], axis=(2, 3))
        
        return Z

    def backward(self, dLdZ):
        """
        Argument:
            dLdZ (np.array): (batch_size, out_channels, output_width, output_height)
        Return:
            dLdA (np.array): (batch_size, in_channels, input_width, input_height)
        """
        _, _, output_width, output_height = dLdZ.shape
        
        dLdA = np.zeros(self.A_shape)

        for i in range(output_width):
            for j in range(output_height):
                dLdA[:, :, i:(i + self.kernel), j:(j + self.kernel)] += dLdZ[:, :, i, j][:, :, None, None] / (self.kernel * self.kernel)

        return dLdA


class MaxPool2d():

    def __init__(self, kernel, stride):
        self.kernel = kernel
        self.stride = stride

        # Create an instance of MaxPool2d_stride1
        self.maxpool2d_stride1 = MaxPool2d_stride1(kernel)
        self.downsample2d = Downsample2d(stride)

    def forward(self, A):
        """
        Argument:
            A (np.array): (batch_size, in_channels, input_width, input_height)
        Return:
            Z (np.array): (batch_size, out_channels, output_width, output_height)
        """
        # Call Maxpool2d_stride1 forward
        Z = self.maxpool2d_stride1.forward(A)

        # Call downsample2d forward
        Z = self.downsample2d.forward(Z)

        return Z

    def backward(self, dLdZ):
        """
        Argument:
            dLdZ (np.array): (batch_size, out_channels, output_width, output_height)
        Return:
            dLdA (np.array): (batch_size, in_channels, input_width, input_height)
        """
        # Call downsample2d backward
        dLdA_downsampled = self.downsample2d.backward(dLdZ)

        # Call Maxpool2d_stride1 backward
        dLdA = self.maxpool2d_stride1.backward(dLdA_downsampled)

        return dLdA


class MeanPool2d():

    def __init__(self, kernel, stride):
        self.kernel = kernel
        self.stride = stride

        # Create an instance of MaxPool2d_stride1
        self.meanpool2d_stride1 = MeanPool2d_stride1(kernel)
        self.downsample2d = Downsample2d(stride)

    def forward(self, A):
        """
        Argument:
            A (np.array): (batch_size, in_channels, input_width, input_height)
        Return:
            Z (np.array): (batch_size, out_channels, output_width, output_height)
        """
        # Call Meanpool2d_stride1 forward
        Z = self.meanpool2d_stride1.forward(A)

        # Call downsample2d forward
        Z = self.downsample2d.forward(Z)

        return Z

    def backward(self, dLdZ):
        """
        Argument:
            dLdZ (np.array): (batch_size, out_channels, output_width, output_height)
        Return:
            dLdA (np.array): (batch_size, in_channels, input_width, input_height)
        """
        # Call downsample2d backward
        dLdA_downsampled = self.downsample2d.backward(dLdZ)

        # Call Meanpool2d_stride1 backward
        dLdA = self.meanpool2d_stride1.backward(dLdA_downsampled)

        return dLdA
