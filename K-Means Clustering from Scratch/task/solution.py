import numpy as np
from sklearn.datasets import load_wine
from matplotlib import pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler

CENTERS = 2
EPS = 1e-6


def plot_comparison(data: np.ndarray, predicted_clusters: np.ndarray, true_clusters: np.ndarray = None,
                    centers: np.ndarray = None, show: bool = True):
    # Use this function to visualize the results on Stage 6.

    if true_clusters is not None:
        plt.figure(figsize=(20, 10))

        plt.subplot(1, 2, 1)
        sns.scatterplot(x=data[:, 0], y=data[:, 1], hue=predicted_clusters, palette='deep')
        if centers is not None:
            sns.scatterplot(x=centers[:, 0], y=centers[:, 1], marker='X', color='k', s=200)
        plt.title('Predicted clusters')
        plt.xlabel('alcohol')
        plt.ylabel('malic_acid')
        plt.grid()

        plt.subplot(1, 2, 2)
        sns.scatterplot(x=data[:, 0], y=data[:, 1], hue=true_clusters, palette='deep')
        if centers is not None:
            sns.scatterplot(x=centers[:, 0], y=centers[:, 1], marker='X', color='k', s=200)
        plt.title('Ground truth')
        plt.xlabel('alcohol')
        plt.ylabel('malic_acid')
        plt.grid()
    else:
        plt.figure(figsize=(10, 10))
        sns.scatterplot(x=data[:, 0], y=data[:, 1], hue=predicted_clusters, palette='deep')
        if centers is not None:
            sns.scatterplot(x=centers[:, 0], y=centers[:, 1], marker='X', color='k', s=200)
        plt.title('Predicted clusters')
        plt.xlabel('alcohol')
        plt.ylabel('malic_acid')
        plt.grid()

    plt.savefig('Visualization.png', bbox_inches='tight')
    if show:
        plt.show()


def calculate_distances(X_full: np.ndarray):
    centers = X_full[:3]
    distances = np.sum((X_full[:, np.newaxis] - centers) ** 2, axis=2)
    nearest_center_indices = np.argmin(distances, axis=1)

    new_centers = np.array([X_full[nearest_center_indices == i].mean(axis=0) for i in range(centers.shape[0])])

    print(new_centers.flatten().tolist())


class CustomKMeans:
    def __init__(self, data, k=CENTERS):
        self.k = k
        self.data = data
        self.centers = data[:k]

    def fit(self, eps=EPS):

        while True:
            distances = np.sum((self.data[:, np.newaxis] - self.centers) ** 2, axis=2)
            nearest_center_indices = np.argmin(distances, axis=1)
            new_centers = np.array(
                [self.data[nearest_center_indices == i].mean(axis=0) for i in range(self.centers.shape[0])])
            squared_distances = np.linalg.norm(self.centers - new_centers, axis=1) ** 2
            if not np.max(squared_distances) >= eps:
                break
            self.centers = new_centers

        return self

    def predict(self):
        distances = np.sum((self.data[:10, np.newaxis] - self.centers) ** 2, axis=2)
        nearest_center_indices = np.argmin(distances, axis=1)
        print(nearest_center_indices.tolist())


def main():
    data = load_wine(as_frame=True, return_X_y=True)
    # print(data)
    X_full, y_full = data

    rnd = np.random.RandomState(42)
    permutations = rnd.permutation(len(X_full))
    X_full = X_full.iloc[permutations]
    y_full = y_full.iloc[permutations]

    X_full = X_full.values
    y_full = y_full.values

    scaler = StandardScaler()
    X_full = scaler.fit_transform(X_full)

    # calculate_distances(X_full)

    CustomKMeans(X_full).fit().predict()


if __name__ == '__main__':
    main()
