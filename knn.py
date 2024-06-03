# KNN: given a data point calc its distance from all other points and return the k nearest points
import numpy as np
from collections import Counter


def euclidean_distance(x1, x2):
    return np.sqrt(np.sum(x1 - x2) ** 2)


class KNN:
    """
    K-nearest neighbors classifier
    """

    def __init__(self, k=3):
        self.k = k

    def fit(self, X, y):
        self.X_train = X
        self.y_train = y
        pass

    def predict(self, X):
        return [self._predict(x) for x in X]

    def _predict(self, x):
        # compute distances
        distances = [euclidean_distance(x, x_train)
                     for x_train in self.X_train]

        # get indices & labels of closest k neighbors
        k_indices = np.argsort(distances)[:self.k]
        k_nearest_labels = [self.y_train[i] for i in k_indices]

        # get the majority vote
        most_common = Counter(k_nearest_labels).most_common()
        return most_common[0][0]
