from sklearn.decomposition import PCA
from sklearn.datasets import load_iris
import matplotlib.pyplot as plt

# Load dataset
data = load_iris()
X = data.data

# Initialize the PCA model
pca = PCA(n_components=2)

# Fit and transform the data
X_pca = pca.fit_transform(X)

# Plot the data
plt.scatter(X_pca[:, 0], X_pca[:, 1], c=data.target)
plt.title('PCA Visualization')
plt.show()
