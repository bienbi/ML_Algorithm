import numpy as np

class SoftmaxRegression:
    def __init__(self, learning_rate = 0.01, epochs=1000):
        self.learning_rate = learning_rate
        self.epochs = epochs

    def softmax(self, X):
        exps = np.exp(X - np.max(X))
        return exps / np.sum(exps, axis=0, keepdims=True)

    def loss_func(self, Y, Y_hat):
        L_sum = np.sum(np.multiply(np.log(Y_hat), Y))
        m = Y.shape[1]
        L = -(1/m) * L_sum
        return L

    def fit(self, X, y):
        dx, N = X.shape
        dy = y.shape[0]

        w = np.random.randn(dy, dx) * 0.01
        b = np.zeros((dy, 1))

        for i in range(self.epochs):
            Z = np.matmul(w, X) + b
            A = self.softmax(Z)

            cost = self.loss_func(A, y)

            dZ = A-y
            dW = (1./N) * np.matmul(dZ, X.T)
            db = (1./N) * np.sum(dZ, axis=1)

            w -= self.learning_rate * dW
            b -= self.learning_rate * db

            if (i % 100 == 0):
                print("Epoch", i, "cost: ", cost)

    def predict(self, X):
        return self.softmax(np.matmul(self.w, X) + self.b)