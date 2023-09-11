import numpy as np

class KNN:
    """
        k: The number of closest neighbors that will determine the class of the sample that we wish to predict.
    """
    def __init__(self, k):
        self.k = k

    def fit(self, X, y):
        self.X = X
        self.y = y

    def predict(self, X):
        # y_pred = []
        # for x in X:
        #     y_pred.append(self._predict(x))
        y_pred = [self._predict(x) for x in X]
        return np.array(y_pred)
        
    def Euclidean_distance(self, x1, x2):
        return np.sqrt(np.sum((x1 - x2)**2))
    
    # Return the most common class among the neighbor sample
    def _predict(self, x):
        distances = [self.Euclidean_distance(x, x_train) for x_train in self.X]
        k_indices = np.argsort(distances)[:self.k]
        k_nearest_labels = [self.y[i] for i in k_indices]
        most_common = np.argmax(np.bincount(k_nearest_labels))
        return most_common