import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from scipy.spatial.distance import pdist, squareform
from sklearn.decomposition import PCA


def main():
    embeddings_dir = Path(sys.argv[1])

    mean_embeddings = []
    _embeddings = {}
    labels = []
    for emb in embeddings_dir.glob("*.npy"):
        _embeddings[emb.stem] = np.load(emb)
    embeddings = np.concatenate(list(_embeddings.values()))
    mean_embeddings = np.stack([a.mean(0) for a in _embeddings.values()])
    labels = [k for k, v in _embeddings.items() for _ in range(len(v))]

    print(squareform(pdist(mean_embeddings)))

    pca = PCA(2)
    embeddings = pca.fit_transform(embeddings)
    mean_embeddings = pca.transform(mean_embeddings)

    plt.figure()
    sns.scatterplot(x=embeddings[:, 0], y=embeddings[:, 1], hue=labels, s=10)
    plt.figure()
    sns.scatterplot(x=mean_embeddings[:, 0], y=mean_embeddings[:, 1], hue=list(_embeddings))
    plt.show()


if __name__ == "__main__":
    main()
