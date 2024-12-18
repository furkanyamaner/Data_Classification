from sklearn.cluster import KMeans
from sklearn.datasets import load_iris
import matplotlib.pyplot as plt

# Load dataset
data = load_iris()
X = data.data

# Initialize the model
kmeans = KMeans(n_clusters=3)

# Train the model
kmeans.fit(X)

# Cluster centers and labels
centers = kmeans.cluster_centers_
labels = kmeans.labels_

# Plot the clusters
plt.scatter(X[:, 0], X[:, 1], c=labels)
plt.scatter(centers[:, 0], centers[:, 1], marker='X', c='red')
plt.title('K-Means Clustering')
plt.show()
