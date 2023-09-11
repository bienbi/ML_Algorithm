import numpy as np
import Decision_Tree

class RandomForest():
    def __init__(self, n_trees=10, max_depth=None):
        self.n_trees = n_trees
        self.max_depth = max_depth
        self.trees = []

    def fit(self, X, y):
        for _ in range(self.n_trees):
            tree = DecisionTree(max_depth=self.max_depth)
            # Bootstrap sample
            indices = np.random.choice(len(X), size=len(X), replace=True)
            X_sample, y_sample = X[indices], y[indices]
            tree.fit(X_sample, y_sample)
            self.trees.append(tree)

    def predict(self, X):
        tree_preds = np.array([tree.predict(X) for tree in self.trees])
        tree_preds = np.swapaxes(tree_preds, 0, 1)      # swapaxes: hoán đổi 2 cột của 1 mảng
        return np.array([np.bincount(tree_pred).argmax() for tree_pred in tree_preds])