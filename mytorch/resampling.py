import numpy as np


class Upsample1d():

    def __init__(self, upsampling_factor):
        self.upsampling_factor = upsampling_factor

    def forward(self, A):
        """
        Argument:
            A (np.array): (batch_size, in_channels, input_width)
        Return:
            Z (np.array): (batch_size, in_channels, output_width)
        """
        # Extract shape information from input
        batch_size, in_channels, input_width = A.shape
        # Calculate output width based on upsampling factor
        output_width = self.upsampling_factor * (input_width - 1) + 1
        
        # Initialize output array with zeros
        Z = np.zeros((batch_size, in_channels, output_width))
        
        # Perform upsampling operation
        Z[:, :, ::self.upsampling_factor] = A
        
        return Z

    def backward(self, dLdZ):
        """
        Argument:
            dLdZ (np.array): (batch_size, in_channels, output_width)
        Return:
            dLdA (np.array): (batch_size, in_channels, input_width)
        """
        # Extract shape information from input gradient
        batch_size, in_channels, output_width = dLdZ.shape
        # Calculate input width based on upsampling factor
        input_width = ((output_width - 1) // self.upsampling_factor) + 1
        
        # Initialize input gradient array
        dLdA = np.zeros((batch_size, in_channels, input_width))
        
        # Calculate gradient with respect to input
        dLdA = dLdZ[:, :, ::self.upsampling_factor]
        
        return dLdA


class Downsample1d():
    def __init__(self, downsampling_factor):
        self.downsampling_factor = downsampling_factor

    def forward(self, A):
        """
        Argument:
            A (np.array): (batch_size, in_channels, input_width)
        Return:
            Z (np.array): (batch_size, in_channels, output_width)
        """
        # Extract shape information from input
        batch_size, in_channels, input_width = A.shape
        # Calculate output width based on downsampling factor
        output_width = ((input_width - 1) // self.downsampling_factor) + 1

        # Initialize output array and perform downsampling operation
        Z = np.zeros((batch_size, in_channels, output_width))
        Z = A[:, :, ::self.downsampling_factor]
        
        # Store input array for use in backward operation
        self.A = A

        return Z

    def backward(self, dLdZ):
        """
        Argument:
            dLdZ (np.array): (batch_size, in_channels, output_width)
        Return:
            dLdA (np.array): (batch_size, in_channels, input_width)
        """
        # Calculate input width based on stored input array
        input_width = self.A.shape[2]

        batch_size, in_channels, _ = dLdZ.shape

        # Initialize input gradient array
        dLdA = np.zeros((batch_size, in_channels, input_width))

        # Calculate gradient with respect to input
        dLdA[:, :, ::self.downsampling_factor] = dLdZ

        return dLdA


class Upsample2d():
    def __init__(self, upsampling_factor):
        self.upsampling_factor = upsampling_factor

    def forward(self, A):
        """
        Argument:
            A (np.array): (batch_size, in_channels, input_height, input_width)
        Return:
            Z (np.array): (batch_size, in_channels, output_height, output_width)
        """
        # Extract shape information from input
        batch_size, in_channels, input_height, input_width = A.shape
        
        # Calculate output dimensions based on upsampling factor
        output_height = self.upsampling_factor * (input_height - 1) + 1
        output_width = self.upsampling_factor * (input_width - 1) + 1

        # Initialize output array with zeros and perform upsampling operation
        Z = np.zeros((batch_size, in_channels, output_height, output_width))
        Z[:, :, ::self.upsampling_factor, ::self.upsampling_factor] = A

        return Z

    def backward(self, dLdZ):
        """
        Argument:
            dLdZ (np.array): (batch_size, in_channels, output_height, output_width)
        Return:
            dLdA (np.array): (batch_size, in_channels, input_height, input_width)
        """
        # Extract shape information from input gradient
        batch_size, in_channels, output_height, output_width = dLdZ.shape
        
        # Calculate input dimensions based on upsampling factor
        input_height = ((output_height - 1) // self.upsampling_factor) + 1
        input_width = ((output_width - 1) // self.upsampling_factor) + 1

        # Initialize input gradient array and calculate gradient with respect to input
        dLdA = np.zeros((batch_size, in_channels, input_height, input_width))
        dLdA = dLdZ[:, :, ::self.upsampling_factor, ::self.upsampling_factor]

        return dLdA

class Downsample2d():
    def __init__(self, downsampling_factor):
        self.downsampling_factor = downsampling_factor

    # Perform forward operation for downsampling
    def forward(self, A):
        """
        Argument:
            A (np.array): (batch_size, in_channels, input_height, input_width)
        Return:
            Z (np.array): (batch_size, in_channels, output_height, output_width)
        """
        # Extract shape information from input
        batch_size, in_channels, input_height, input_width = A.shape
        
        # Calculate output dimensions based on downsampling factor
        output_height = ((input_height - 1) // self.downsampling_factor) + 1
        output_width = ((input_width - 1) // self.downsampling_factor) + 1

        # Perform downsampling operation
        Z = np.zeros((batch_size, in_channels, output_height, output_width))
        Z = A[:, :, ::self.downsampling_factor, ::self.downsampling_factor]

        # Store input array for use in backward operation
        self.A = A

        return Z

    def backward(self, dLdZ):
        """
        Argument:
            dLdZ (np.array): (batch_size, in_channels, output_height, output_width)
        Return:
            dLdA (np.array): (batch_size, in_channels, input_height, input_width)
        """
        # Extract shape information for input dimensions from stored input array
        batch_size, in_channels, _, _ = dLdZ.shape
        _, _, input_height, input_width = self.A.shape

        # Initialize input gradient array and calculate gradient with respect to input
        dLdA = np.zeros((batch_size, in_channels, input_height, input_width))
        dLdA[:, :, ::self.downsampling_factor, ::self.downsampling_factor] = dLdZ

        return dLdA
