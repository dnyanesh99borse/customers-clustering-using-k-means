🛒 Customer Segmentation Using K-Means Clustering
🚀 A machine learning project to group retail customers based on their purchase behavior using K-Means clustering—built completely from scratch!

📌 Introduction
Understanding customer behavior is critical for any business. This project utilizes K-Means Clustering to group customers into different segments based on their Annual Income and Spending Score.
✅ Helps businesses identify customer segments
✅ Allows targeted marketing & promotions
✅ Improves inventory planning & recommendations

🔍 What is Clustering?
Clustering is an unsupervised machine learning technique used to group similar data points together.
🏆 K-Means Clustering
K-Means divides the dataset into K groups based on feature similarity. It follows these steps: 1️⃣ Select the number of clusters (K)
2️⃣ Initialize K random centroids
3️⃣ Assign each customer to the closest centroid
4️⃣ Compute the new centroid positions
5️⃣ Repeat until centroids stop moving
Example: Grouping customers based on their spending habits helps retailers design better marketing strategies! 🎯


🛠️ Project Implementation
🔹 Tech Stack
- Python 🐍
- Numpy & Pandas (Data handling)
- Matplotlib (Data visualization)
📦 Required Libraries
pip install numpy pandas matplotlib



📂 Dataset
We use a dataset containing customer purchase behavior, with columns: | CustomerID | Gender | Age | Annual Income (k$) | Spending Score (1-100) | |--------------|------------|----|-----------------|--------------------| | 1          | Male       | 19  | 15             | 39                 | | 2          | Male       | 21  | 15             | 81                 |
Key Features Used: Annual Income, Spending Score
These help us cluster customers based on their spending habits!


🔧 Step-by-Step Guide to Building K-Means from Scratch
1️⃣ Import Necessary Libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


2️⃣ Load the Dataset
df = pd.read_csv("customer_data.csv")
X = df[['Annual Income (k$)', 'Spending Score (1-100)']].values


3️⃣ Implement K-Means Clustering (From Scratch)
Initialize Centroids
def initialize_centroids(X, K):
    np.random.seed(42)
    return X[np.random.choice(X.shape[0], K, replace=False)]


Compute Euclidean Distance
def euclidean_distance(point1, point2):
    return np.sqrt(np.sum((point1 - point2) ** 2))


Assign Clusters
def assign_clusters(X, centroids):
    clusters = []
    for point in X:
        distances = [euclidean_distance(point, centroid) for centroid in centroids]
        cluster = np.argmin(distances)
        clusters.append(cluster)
    return np.array(clusters)


Update Centroids
def update_centroids(X, clusters, K):
    new_centroids = []
    for k in range(K):
        cluster_points = X[clusters == k]
        new_centroid = np.mean(cluster_points, axis=0)
        new_centroids.append(new_centroid)
    return np.array(new_centroids)


Run K-Means Algorithm
def k_means_clustering(X, K, max_iterations=100):
    centroids = initialize_centroids(X, K)
    
    for _ in range(max_iterations):
        clusters = assign_clusters(X, centroids)
        new_centroids = update_centroids(X, clusters, K)

        if np.all(centroids == new_centroids):
            break
        centroids = new_centroids

    return centroids, clusters


Execute with K=3 Clusters
K = 3
final_centroids, cluster_assignments = k_means_clustering(X, K)
print("\nFinal Centroids:\n", final_centroids)
print("\nCustomer Cluster Assignments:\n", cluster_assignments)



📊 Visualizing the Clusters
plt.scatter(X[:, 0], X[:, 1], c=cluster_assignments, cmap='viridis', marker='o')
plt.scatter(final_centroids[:, 0], final_centroids[:, 1], color='red', marker='X', s=200, label="Centroids")
plt.xlabel("Annual Income (Normalized)")
plt.ylabel("Spending Score (Normalized)")
plt.title("Customer Segmentation Using K-Means")
plt.legend()
plt.show()



🎯 Results & Business Insights
🔹 Understanding Customer Groups
- Cluster 1: Luxury buyers 💎 → High income, High spending
- Cluster 2: Budget-conscious customers 💰 → Low spending score
- Cluster 3: Moderate spenders 🎯 → Balanced spending
🔹 Real-Life Usefulness
✔ Targeted marketing → Personalized discounts per group
✔ Smart inventory management → Stock popular items for each segment
✔ Better customer engagement → Send tailored promotions

🌟 Future Improvements
- ✅ Add more features (Age, Gender, Purchase Frequency)
- ✅ Use Elbow Method to optimize K value
- ✅ Integrate a dashboard (Streamlit/Flask) to analyze customers dynamically

✨ Final Thoughts
With this K-Means clustering model, businesses can gain valuable customer insights and make data-driven decisions. The project:
- Segments customers efficiently based on purchase behavior
- Creates personalized marketing strategies for businesses
- Improves inventory planning & recommendation systems
🚀 Ready to take this project to the next level? Integrate it with a live web dashboard for real-time analysis! 💡

🤝 Contributors
🙌 Built by Dnyaneshwar Sambhaji Borse Open for collaboration
🔗 Feel free to improve, extend, or contribute to this project!
License: MIT


This README is detailed, informative, and visually engaging—perfect for showcasing your project professionally! 🚀💡 Let me know if you’d like any refinements or additional elements. 😊🎯
