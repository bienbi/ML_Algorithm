class SVM():
    def __init__(self, learning_rate = 0.01, C = 100, max_iters = 1000):
        self.lr = learning_rate
        self.C = C
        self.max_iters = max_iters

    def fit(self, X, y):
        w = np.zeros(X.shape[1])
        b = 0
        y = np.where(y >0, 1, -1)
        for _ in range(self.max_iters):
            for i in range(X.shape[0]):
                cond = y[i] * (self.linear_kernel(X[i], w) + b)
                if cond >= 1:
                    w -= self.lr * 1 / self.C * w
                else:
                    w -= self.lr * (1*self.lr - np.dot(X[i], y[i]))
                    b += y[i]
        self.w = w
        self.b = b
        return w, b
    
    def linear_kernel(self, x1, x2):
        return np.dot(x1, x2)
    
    def gaussian_kernel(self, x1, x2):
        gamma = 0.1
        return np.exp(-gamma * np.linalg.norm(x1-x2)**2)
    
    def polynomial_kernel(self, x1, x2):
        coef0 = 1
        degree = 3
        gamma = 0.1
        return (coef0 + gamma * np.dot(x1, x2))**degree
    
    def sigmoid_kernel(self, x1, x2):
        coef0 = 1
        gamma = 0.1
        return np.tanh(coef0 + gamma * np.dot(x1, x2))

    def predict(self, X):
        return np.sign(np.dot(X, self.w) + self.b)