import numpy as np

class LogisticRegression():
    def __init__(self, regular_param = 0.001, lr = 0.05, epoches = 1000, threshold = 0.5):
        """
            - regular_param - regularization parameter
            - lr - learning rate: The step length that will be taken when following the negative gradient during training.
        """
        self.regular_param = regular_param      
        self.lr = lr                            
        self.epoches = epoches
        self.threshold = threshold

    def sigmoid(self, S):
        return 1/(1 + np.exp(-S))

    # Estimate the output probability
    def prob(self, w, X):
        return self.sigmoid(X.dot(w))

    # Loss function with weight decay
    def loss(self, w, X, y):
        z = self.prob(w, X)
        return -np.mean(y*np.log(z) + (1-y)*np.log(1-z)) + 0.5*self.regular_param/X.shape[0]*np.sum(w*w)

    def fit(self, X, y):
        X_b = np.concatenate((X, np.ones((X.shape[0], 1))), axis = 1) 
        N, d = X_b.shape 
        w_init = np.random.randn(d)
        w = w_old = w_init 

        # store history of loss in loss_hist
        loss_hist = [self.loss(w_init, X_b, y)]
        
        ep = 0 
        while ep < self.epoches: 
            ep += 1
            mix_ids = np.random.permutation(N) # Shuffle 
            for i in mix_ids:
                xi = X_b[i]
                yi = y[i]
                zi = self.sigmoid(xi.dot(w))
                # update w
                w  -= self.lr*((zi - yi)*xi + self.regular_param*w)

            loss_hist.append(self.loss(w, X_b, y))
            # Condition to stop 
            if np.linalg.norm(w - w_old)/d < 1e-6:
                break 
            w_old = w
        self.w = w
        return w, loss_hist 
        
    def predict(self, X): 
        X_b = np.concatenate((X, np.ones((X.shape[0], 1))), axis = 1) 
        res = np.zeros(X_b.shape[0])
        res[np.where(self.prob(self.w, X_b) > self.threshold)[0]] = 1
        return res 
