import numpy as np

class KMeans():
    def __init__(self, n_clusters):
        self.n_clusters = n_clusters

   
    def fit(self, X):
        #self.cluster_center = X[np.random.choice(X.shape[0], self.n_clusters, replace=False)]
        old_cluster_center = self.cluster_init(X, self.n_clusters)

        while True:
            labels = self.cluster_labels(X, old_cluster_center)

            new_cluster_center = self.cluster_update(X, labels, self.n_clusters)

            if np.all(old_cluster_center == new_cluster_center):
                self.cluster_center = new_cluster_center
                return new_cluster_center
            old_cluster_center = new_cluster_center


    # Khởi tạo tâm cụm bằng cách chọn ngẫu nhiên n_clusters điểm trong tập X
    def cluster_init(self, X, n_clusters):
        return  X[np.random.choice(X.shape[0], n_clusters, replace=False)]
    
    # Tìm nhãn mới cho các điểm khi biết tâm cụm
    def cluster_labels(self, X, cluster_center):
        return [self.nearest(cluster_center, x) for x in X] 
    
    # Tính khoảng cách Euclid giữa 2 điểm a và b
    def Euclidean_distance(self, a, b):
        return np.sqrt(((a - b)**2).sum())
    
    # Trả về chỉ số của cụm gần dữ liệu nhất
    def nearest(self, clusters, x):
        return np.argmin([self.Euclidean_distance(x, c) for c in clusters]) # tìm nhãn mới cho các điểm khi biết các tâm cụm

     # cập nhật các tâm cụm mới khi biết nhãn của từng điểm
    def cluster_update(self, X, labels, n_clusters):
        cluster_centers_update = np.zeros([n_clusters, X.shape[1]])
        for i in range(n_clusters):
            cluster_centers_update[i, :] = np.mean(X[labels] == i, axis = 0)
        return cluster_centers_update
   
   
    def predict(self, X):
        return self.cluster_labels(X, self.cluster_center)
