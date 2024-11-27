import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score
import numpy as np

# Step 1: Load the dataset
df = pd.read_csv("I:\\My Drive\\DWDM\\assignments\\assignment_7\\lung cancer survey.csv")

# Step 2: Preprocessing (Convert categorical to numerical)
df['LUNG_CANCER'] = df['LUNG_CANCER'].map({'YES': 1, 'NO': 0})

# Drop non-numeric columns
features = df.drop(columns=['LUNG_CANCER', 'GENDER'])  # Exclude non-numeric columns if any

# Step 3: Standardize the features
scaler = StandardScaler()
scaled_features = scaler.fit_transform(features)

# Step 4: Apply K-Means Clustering and determine the optimal number of clusters
inertia = []
K = range(1, 11)  # Trying different cluster numbers from 1 to 10

for k in K:
    kmeans = KMeans(n_clusters=k, random_state=42)
    kmeans.fit(scaled_features)
    inertia.append(kmeans.inertia_)

# Plot the Elbow Method graph
plt.figure(figsize=(8, 5))
plt.plot(K, inertia, marker='o')
plt.xlabel('Number of clusters')
plt.ylabel('Inertia')
plt.title('Elbow Method For Optimal k')
plt.show()

# Step 5: Fit KMeans with optimal clusters (assuming 3 clusters)
kmeans = KMeans(n_clusters=3, random_state=42)
df['Cluster'] = kmeans.fit_predict(scaled_features)

# Step 6: Visualize the clusters with PCA reduction
pca = PCA(n_components=2)
pca_features = pca.fit_transform(scaled_features)
pca_df = pd.DataFrame(pca_features, columns=['PC1', 'PC2'])
pca_df['Cluster'] = df['Cluster']

plt.figure(figsize=(10, 6))
sns.scatterplot(data=pca_df, x='PC1', y='PC2', hue='Cluster', palette='viridis', s=100, alpha=0.7)
plt.title('K-Means Clustering with PCA Reduction')
plt.xlabel('Principal Component 1')
plt.ylabel('Principal Component 2')
plt.legend(title='Cluster')
plt.show()

# Print cluster characteristics
labels = kmeans.labels_
print("Cluster Centroids:")
print(kmeans.cluster_centers_)

print("Inertia:", kmeans.inertia_)

unique, counts = np.unique(labels, return_counts=True)
cluster_sizes = dict(zip(unique, counts))
print("Cluster Sizes:")
print(cluster_sizes)

sil_score = silhouette_score(scaled_features, labels)
print("Silhouette Score:", sil_score)

# Display the DataFrame with cluster assignments
print(df.head())
