import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv("customers.csv")  # Replace with your actual file name
print(df.head())  # View the first few rows

X = df[['Annual_Income_(k$)','Spending_Score_(1-100)']].values
X = (X-np.mean(X, axis = 0)) / np.std(X, axis=0)#For Normalising the data,Scaling helps improve clustering accuracy

def initialize_centroids(X, K):
    np.random.seed(42) #ensure reproductivity

    return X[np.random.choice(X.shape[0],K,replace=False)]

def euclidean_distance(point1, point2):
    return np.sqrt(np.sum((point1 - point2) ** 2))

def assign_clusters(X, centroids):
    clusters = []
    for point in X:
        distances = [euclidean_distance(point, centroid) for centroid in centroids]
        cluster = np.argmin(distances)  # Nearest centroid
        clusters.append(cluster)
    return np.array(clusters)

def update_centroids(X, clusters, K):
    new_centroids = []
    for k in range(K):
        cluster_points = X[clusters == k]
        new_centroid = np.mean(cluster_points, axis=0)
        new_centroids.append(new_centroid)
    return np.array(new_centroids)

def k_means_clustering(X, K, max_iterations=100):
    # Step 1: Initialize centroids
    centroids = initialize_centroids(X, K)
    
    for _ in range(max_iterations):
        # Step 2: Assign clusters
        clusters = assign_clusters(X, centroids)
        
        # Step 3: Compute new centroids
        new_centroids = update_centroids(X, clusters, K)
        
        # Step 4: Check for convergence
        if np.all(centroids == new_centroids):  # Stop if centroids don't change
            break
        centroids = new_centroids
    
    return centroids, clusters

K = 3  # Number of customer segments
final_centroids, cluster_assignments = k_means_clustering(X, K)

def spending_level(score):
    if score <= 40:
        return 'Low Spender'
    elif score <= 70:
        return 'Moderate Spender'
    else:
        return 'High Spender'
df['Spending Category'] = df['Spending_Score_(1-100)'].apply(spending_level)

spending_counts = df['Spending Category'].value_counts()
print(spending_counts)


print("\nFinal Centroids:\n", final_centroids)
print("\nCustomer Cluster Assignments:\n", cluster_assignments)

plt.scatter(X[:, 0], X[:, 1], c=cluster_assignments, cmap='viridis', marker='o')
plt.scatter(final_centroids[:, 0], final_centroids[:, 1], color='red', marker='X', s=200, label="Centroids")
plt.xlabel("Annual_Income (Normalized)")
plt.ylabel("Spending_Score (Normalized)")
plt.title("Customer Segmentation Using K-Means")
plt.legend()
plt.show()


spending_counts.plot(kind='bar', color=['skyblue', 'orange', 'green'])
plt.title("Customer Spending Distribution")
plt.ylabel("Number of Customers")
plt.xlabel("Spending Category")
plt.show()

print("\n")

print("- Cluster 0: has moderate-to-high income, low spending customers.")
print("- Cluster 1: includes mostly low income and medium spending customers — which seems to be the largest group in your dataset.")
print("- Cluster 2:  includes high-income, high-spending customers — potentially your VIPs!")