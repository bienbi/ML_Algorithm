import numpy as np

class NaiveBayes():
    def __init__(self):
        pass

    def fit(self, X, y):
        # Phân chia dữ liệu thành các nhóm dựa trên nhãn của chúng
        split = []                  # Chứa các nhóm dữ liệu
        for c in np.unique(y):      # Lặp qua tất cả các giá trị duy nhất của y
            group = []          # chứa các giá trị của X có nhãn là 'c'
            for x, t in zip(X, y):  # zip(): kết hợp các phần tử theo cặp
                if t == c:
                    group.append(x)
            split.append(group)
        #split = [[x for x, t in zip(X, y) if t == c] for c in np.unique(y)]

        summarize = []                  # chứa các mảng là độ lệch chuẩn và giá trị trung bình của mỗi nhóm dữ liệu trong 'split'
        for i in split:
            mean = np.mean(i, axis=0)
            std = np.std(i, axis=0)
            summarize.append(np.c_[mean, std])  # np.c_: nối hai mảng theo cột
        self.summarize = summarize
        #self.model = np.array([np.c_[np.mean(i, axis=0), np.std(i, axis=0)] for i in split])
        
        return self

    def prob(self, x, mean, std):   # Tính phân phối Gauss
        exponent = np.exp(- ((x - mean)**2 / (2 * std**2)))
        return np.log(exponent / (np.sqrt(2 * np.pi) * std))

    def predict_prob(self, X):   # Dự đoán xác suất cho mỗi điểm dữ liệu của từng nhóm
        result = []
        for x in X:
            proba = []
            for i in self.summarize:
                log_proba = 0   # Tính tổng xác suất log cho mỗi mẫu dữ liệu
                for s, j in zip(i, x):
                    log_proba += self.prob(j, *s)
                proba.append(log_proba)
            result.append(proba)
        return result
        #return [[sum(self.prob(i, *s) for s, i in zip(summaries, x)) for summaries in self.model] for x in X]

    def predict(self, X):
        return np.argmax(self.predict_prob(X), axis=1)
