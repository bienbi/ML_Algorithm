import numpy as np

class Perceptron():
    def __init__(self):
        pass
    
    def fit(self, X, y):
        X_b = np.concatenate((np.ones((X.shape[0], 1)), X), axis = 1)
        w = np.random.randn(X_b.shape[1], 1)

        while True:
            pred = np.sign(np.dot(X_b, w))
            # find indexes of misclassified points
            mis_idxs = np.where(np.equal(pred, y) == False)[0]
            # number of misclassified points
            num_mis = mis_idxs.shape[0]
            if num_mis == 0: # no more misclassified points
                return w
            # randomly pick one misclassified point
            random_id = np.random.choice(mis_idxs, 1)[0]
            # update w
            w += X_b[random_id] * y[random_id]

        self.w = w
        return w

    def predict(self, X):
        X_b = np.concatenate((np.ones((X.shape[0], 1)), X), axis = 1)
        return np.sign(np.dot(X_b, self.w))