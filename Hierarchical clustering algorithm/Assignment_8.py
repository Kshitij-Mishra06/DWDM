import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.cluster.hierarchy import dendrogram, linkage
from sklearn.preprocessing import LabelEncoder, StandardScaler
import numpy as np
from sklearn.cluster import AgglomerativeClustering

# Load the dataset (assuming the dataset is in a CSV format)
df = pd.read_csv('I:/My Drive/DWDM/assignments/Assignment_8/user_based_sample_dataset.csv')

# Display the first few rows of the dataset
print(df.head())

# Preprocessing: Convert categorical variables to numeric using Label Encoding
label_encoder = LabelEncoder()
df['GENDER'] = label_encoder.fit_transform(df['GENDER'])  # Encode 'M' and 'F'
df['LUNG_CANCER'] = label_encoder.fit_transform(df['LUNG_CANCER'])  # Encode 'YES' and 'NO'

# Feature scaling (standardize features before clustering)
scaler = StandardScaler()
X_scaled = scaler.fit_transform(df.drop(columns=['LUNG_CANCER']))  # All columns except target

# Perform hierarchical/agglomerative clustering
linked = linkage(X_scaled, method='ward')  # 'ward' minimizes variance within clusters

# Plotting Dendrogram
plt.figure(figsize=(10, 7))
dendrogram(linked,
           orientation='top',
           labels=np.array(df['LUNG_CANCER']),
           distance_sort='descending',
           show_leaf_counts=True)
plt.title("Dendrogram for Hierarchical Clustering")
plt.xlabel("Sample Index")
plt.ylabel("Distance (Euclidean)")
plt.xticks(ticks=np.arange(len(df)), labels=df.index, rotation=90)  # X-ticks as sample indices
plt.yticks(fontsize=10)  # Adjust y-ticks font size
plt.show()

# Perform clustering with the desired number of clusters (e.g., 2 for binary classification)
# Remove the 'affinity' argument as it has been deprecated
cluster_model = AgglomerativeClustering(n_clusters=2, linkage='ward')
cluster_labels = cluster_model.fit_predict(X_scaled)

# Scatter Plot: let's visualize AGE vs SMOKING and color by cluster labels
plt.figure(figsize=(8, 6))
sns.scatterplot(x=df['AGE'], y=df['SMOKING'], hue=cluster_labels, palette='deep', s=100, marker='o')
plt.title("Dot Chart: Clusters based on AGE and SMOKING")
plt.xlabel("Age")
plt.ylabel("Smoking Level")
plt.legend(title="Cluster", loc='upper right', labels=['Cluster 0', 'Cluster 1'])  # Custom legend labels
plt.grid(True)
plt.show()
