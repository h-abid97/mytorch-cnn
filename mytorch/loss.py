import numpy as np

# Copy your Linear class from HW1P1 here
class MSELoss:

    def forward(self, A, Y):
        """
        Calculate the Mean Squared error
        :param A: Output of the model of shape (N, C)
        :param Y: Ground-truth values of shape (N, C)
        :Return: MSE Loss(scalar)

        """

        self.A = A
        self.Y = Y
        self.N = A.shape[0]  # TODO
        self.C = A.shape[1]  # TODO
        se = (A - Y) * (A - Y)  # TODO
        sse = np.ones((self.N, 1)).T @ se @ np.ones((self.C, 1))  # TODO
        mse = sse/(2*self.N*self.C)  # TODO

        return mse

    def backward(self):

        dLdA = (self.A - self.Y)/(self.N*self.C)  # TODO

        return dLdA


class CrossEntropyLoss:

    def forward(self, A, Y):
        """
        Calculate the Cross Entropy Loss
        :param A: Output of the model of shape (N, C)
        :param Y: Ground-truth values of shape (N, C)
        :Return: CrossEntropyLoss(scalar)

        Refer the the writeup to determine the shapes of all the variables.
        Use dtype ='f' whenever initializing with np.zeros()
        """
        self.A = A
        self.Y = Y
        N = A.shape[0]  # TODO
        C = A.shape[1]  # TODO

        Ones_C = np.ones((C, 1), dtype = 'f')  # TODO
        Ones_N = np.ones((N, 1), dtype = 'f')  # TODO

        self.softmax = np.exp(A)/np.exp(A).sum(axis=1, keepdims=True)  # TODO
        crossentropy = (-1*self.Y * np.log(self.softmax)) @ Ones_C # TODO
        sum_crossentropy = Ones_N.T @ crossentropy # TODO
        L = sum_crossentropy / N

        return L

    def backward(self):

        dLdA = self.softmax - self.Y  # TODO

        return dLdA
