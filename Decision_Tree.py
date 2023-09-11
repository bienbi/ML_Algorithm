import numpy as np

class Node:
    def __init__(self, predicted_class):
        self.predicted_class = predicted_class
        self.feature_index = 0
        self.threshold = 0
        self.left = None
        self.right = None

class DecisionTree():
    def __init__(self, max_depth=None):
        self.max_depth = max_depth

    def fit(self, X, y):
        self.n_classes = len(set(y))
        self.n_features = X.shape[1]
        self.tree = self.grow_tree(X, y)

    def predict(self, X):
        return np.array([self._predict(inputs) for inputs in X])

    def _gini(self, y):
        N = y.size
        gini = 0
        for i in range(self.n_classes):
            # đếm số lượng mẫu trong tập y có giá trị  = i
            class_count = np.sum(y == i)
            prop = class_count / N
            gini += prop ** 2
        gini = 1.0 - gini

        return gini
        # return 1.0 - sum((np.sum(y == c) / N) ** 2 for c in range(self.n_classes))

    def best_split(self, X, y):
        N = y.size
        num_parent = [np.sum(y == c) for c in range(self.n_classes)]   # Chứa số lượng của từng lớp trong y
        best_gini = 1.0 - sum((n / N) ** 2 for n in num_parent)         # Chỉ số gini thấp nhất có thể đạt được
        
        best_idx, best_thr = None, None
        
        for idx in range(self.n_features):
            thresholds, classes = zip(*sorted(zip(X[:, idx], y)))       # Sắp xếp các đặc trưng và các lớp tương ứng
            # # Step 1: Ghép nối các đặc trưng với lớp tương ứng
            # feature_values = X[:, idx]
            # class_labels = y
            # pairs = list(zip(feature_values, class_labels))

            # # Step 2: Sắp xếp các cặp theo đặc trưng
            # sorted_pairs = sorted(pairs, key=lambda pair: pair[0])

            # # Step 3: Giải nén các cặp đã sắp xếp thành hai danh sách
            # thresholds, classes = zip(*sorted_pairs)

            num_left = [0] * self.n_classes
            num_right = num_parent.copy()

            # Lặp qua từng ngưỡng
            for i in range(1, N):
                c = classes[i - 1]
                num_left[c] += 1
                num_right[c] -= 1

                # Tính chỉ số Gini cho tập các node con bên trái và bên phải
                gini_left = 1.0 - sum((num_left[x] / i) ** 2 for x in range(self.n_classes))
                gini_right = 1.0 - sum((num_right[x] / (N - i)) ** 2 for x in range(self.n_classes))

                gini = (i * gini_left + (N - i) * gini_right) / N

                if thresholds[i] == thresholds[i - 1]:
                    continue

                # Cập nhật cách chia tốt nhất
                if gini < best_gini:
                    best_gini = gini
                    best_idx = idx
                    best_thr = (thresholds[i] + thresholds[i - 1]) / 2

        return best_idx, best_thr

    def grow_tree(self, X, y, depth=0):
        # Tính số lượng mẫu trên mỗi lớp để xác định lớp dự đoán của nút hiện tại. Lớp có nhiều mẫu nhất được chọn làm lớp dự đoán.
        num_samples_per_class = [np.sum(y == i) for i in range(self.n_classes)]
        predicted_class = np.argmax(num_samples_per_class)

        # Tạo node mới với lớp được dự đoán. Nếu điều kiện dừng được thỏa mãn, nút này trở thành node lá 
        node = Node(predicted_class=predicted_class)

        if depth < self.max_depth:
            # Tìm cách phân tách tốt nhất
            idx, thr = self.best_split(X, y)
            if idx is not None:
                node.feature_index = idx
                node.threshold = thr

                # Phân tách dữ liệu thành 2 tập con dựa trên feature và ngưỡng
                indices_left = X[:, idx] < thr
                X_left, y_left = X[indices_left], y[indices_left]       # nếu feature < ngưỡng, dữu liệu chuyển về tập con bên trái
                X_right, y_right = X[~indices_left], y[~indices_left]   # các trường hợp còn lại thì về bên phải

                # Phát triển cây con
                node.left = self.grow_tree(X_left, y_left, depth + 1)
                node.right = self.grow_tree(X_right, y_right, depth + 1)
                
        return node

    def _predict(self, inputs):
        # Bắt đầu dự đoán tại node gốc
        node = self.tree

        # Duyệt cây và phân chia node con
        while node.left:
            if inputs[node.feature_index] < node.threshold:
                node = node.left
            else:
                node = node.right

        return node.predicted_class