import numpy as np

# Copy your Linear class from HW1P1 here
class Identity:

    def forward(self, Z):

        self.A = Z

        return self.A

    def backward(self):

        dAdZ = np.ones(self.A.shape, dtype="f")

        return dAdZ


class Sigmoid:
    """
    On same lines as above:
    Define 'forward' function
    Define 'backward' function
    Read the writeup for further details on Sigmoid.
    """
    def forward(self, Z):

        self.A = 1/(1 + np.exp(-1*Z))

        return self.A

    def backward(self):

        dAdZ = self.A - (self.A * self.A)

        return dAdZ


class Tanh:
    """
    On same lines as above:
    Define 'forward' function
    Define 'backward' function
    Read the writeup for further details on Tanh.
    """
    def forward(self, Z):

        #self.A = (np.exp(Z) - np.exp(-1*Z))/(np.exp(Z) + np.exp(-1*Z))
        self.A = np.tanh(Z)

        return self.A

    def backward(self):

        dAdZ = 1 - (self.A * self.A)

        return dAdZ

class ReLU:
    """
    On same lines as above:
    Define 'forward' function
    Define 'backward' function
    Read the writeup for further details on ReLU.
    """
    def forward(self, Z):

        self.A = np.maximum(0, Z)

        return self.A

    def backward(self):

        dAdZ = np.where(self.A > 0, 1, 0)

        return dAdZ
