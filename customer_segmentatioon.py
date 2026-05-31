import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

# Load dataset
data = pd.read_csv("customers.csv")

# Select features
X = data[['Age', 'Annual_Income', 'Spending_Score']]

# Scale data
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Apply K-Means Clustering
kmeans = KMeans(n_clusters=4, random_state=42)
data['Cluster'] = kmeans.fit_predict(X_scaled)

# Display cluster counts
print(data['Cluster'].value_counts())

# Visualize Clusters
plt.figure(figsize=(8,6))
plt.scatter(data['Annual_Income'],
            data['Spending_Score'],
            c=data['Cluster'],
            cmap='viridis')

plt.xlabel('Annual Income')
plt.ylabel('Spending Score')
plt.title('Customer Segmentation')
plt.colorbar(label='Cluster')
plt.show()

# Save output
data.to_csv("segmented_customers.csv", index=False)
print("Segmentation completed!")
