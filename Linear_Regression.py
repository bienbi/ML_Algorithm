import numpy as np

class Multiple_Linear_Regression():   
    def __init__ (self):
        self.w = np.zeros([1, 1]); 
    
    def fit(self, X, y):
        X_b = np.c_[np.ones(len(X)), X] 
        w = np.linalg.pinv(X_b.T.dot(X_b)).dot(X_b.T).dot(y)
        self.w = w
    
    def predict(self, X):
        X = np.c_[np.ones((len(X), 1)), X]
        
        return np.dot(X, self.w)