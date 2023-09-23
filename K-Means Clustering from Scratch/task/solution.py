import numpy as np
from sklearn.datasets import load_wine
from matplotlib import pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler

# scroll down to the bottom to implement your solution


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
    last_ten_points = X_full[-10:]
    distances = np.sum((last_ten_points[:, np.newaxis] - centers)**2, axis=2)
    nearest_center_indices = np.argmin(distances, axis=1)
    nearest_center_indices_list = nearest_center_indices.tolist()
    print(nearest_center_indices_list)



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

    calculate_distances(X_full)

if __name__ == '__main__':
    main()

