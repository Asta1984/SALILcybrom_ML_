# #Create synthetic datasets
# import matplotlib.pyplot as plt
# from sklearn.datasets import make_moons, make_circles, make_blobs

# # # 1. make_moons
# X_moons, y_moons = make_moons(n_samples=300, noise=0.05, random_state=42)

# # # 2. make_circles
# X_circles, y_circles = make_circles(n_samples=300, factor=0.5, noise=0.05, random_state=42)

# # # 3. make_blobs
# X_blobs, y_blobs = make_blobs(n_samples=300, centers=3, cluster_std=0.6, random_state=42)

# # # Create subplots
# fig, axes = plt.subplots(1, 3, figsize=(15, 4))

# # # Plot moons
# axes[0].scatter(X_moons[:,0], X_moons[:,1], c=y_moons, cmap="plasma", s=40, edgecolors="k")
# axes[0].set_title("make_moons")

# # # Plot circles
# axes[1].scatter(X_circles[:,0], X_circles[:,1], c=y_circles, cmap="plasma", s=40, edgecolors="k")
# axes[1].set_title("make_circles")

# # # Plot blobs
# axes[2].scatter(X_blobs[:,0], X_blobs[:,1], c=y_blobs, cmap="plasma", s=40, edgecolors="k")
# axes[2].set_title("make_blobs")

# plt.suptitle("Comparison of Synthetic Datasets", fontsize=14)
# plt.show()

#------------------------------------------------------------------------------------------------------------------------------
#model implementation on moon type 

# import numpy as np
# import matplotlib.pyplot as plt
# from sklearn.cluster import DBSCAN
# from sklearn.datasets import make_moons

# # # Step 1: Create sample data (two interleaving half circles)
# X, _ = make_moons(n_samples=300, noise=0.05, random_state=42)

# # # Step 2: Apply DBSCAN
# db = DBSCAN(eps=0.2, min_samples=5)   # eps = neighborhood radius, min_samples = min points per cluster
# db.fit(X)

# # # Step 3: Extract labels
# labels = db.labels_   # -1 means noise
# n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
# print(f"Estimated number of clusters: {n_clusters}")

# # # Step 4: Plot results
# plt.figure(figsize=(6,6))
# plt.scatter(X[:,0], X[:,1], c=labels, cmap="plasma", s=50, edgecolors="k")
# plt.title(f"DBSCAN Clustering (clusters={n_clusters})")
# plt.show()

#-------------------------------------------------------------------------------------------------------
#Density based clustering algoritmn on synthetic concentric circle

# import numpy as np
# import matplotlib.pyplot as plt
# from sklearn.datasets import make_circles
# from sklearn.cluster import DBSCAN

# # Step 1: Generate two-circle dataset
# X, _ = make_circles(n_samples=500, factor=0.5, noise=0.03, random_state=42)

# # Step 2: Different eps values to test
# eps_values = [0.1, 0.2, 0.3, 0.4]

# # Step 3: Create subplots
# fig, axes = plt.subplots(1, len(eps_values), figsize=(16, 4))

# for i, eps in enumerate(eps_values):
#     db = DBSCAN(eps=eps, min_samples=5)
#     labels = db.fit_predict(X)
    
#     n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
    
#     axes[i].scatter(X[:,0], X[:,1], c=labels, cmap="plasma", s=30, edgecolors="k")
#     axes[i].set_title(f"eps={eps}\nclusters={n_clusters}")

# plt.suptitle("DBSCAN with Different eps Values", fontsize=14)
# plt.show()

#-------------------------------------------------------------------------------------------------------

# import numpy as np
# import matplotlib.pyplot as plt
# from sklearn.datasets import make_circles
# from sklearn.cluster import DBSCAN
# from sklearn.preprocessing import StandardScaler

# # Data
# X, y = make_circles(n_samples=500, factor=0.5, noise=0.03, random_state=42)

# print(y)

# # Scale
# X_scaled = StandardScaler().fit_transform(X)

# # DBSCAN (tuned eps)
# db = DBSCAN(eps=0.25, min_samples=5)
# labels = db.fit_predict(X_scaled)

# # Count clusters
# n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
# print(f"Estimated clusters: {n_clusters}")

# # Plot
# plt.scatter(X[:,0], X[:,1], c=labels, cmap="plasma", s=50, edgecolors="k")
# plt.title(f"Improved DBSCAN on Two Circles (clusters={n_clusters})")
# plt.show()

#-------------------------------------------------------------------------------------------------------

import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_circles
from sklearn.cluster import DBSCAN, KMeans

# Step 1: Generate nonlinear "two circles" dataset
X, y = make_circles(n_samples=500, factor=0.5, noise=0.05, random_state=42)

# Step 2: Apply K-Means
kmeans = KMeans(n_clusters=2,init="k-means++", random_state=42)
kmeans_labels = kmeans.fit_predict(X)

# Step 3: Apply DBSCAN
dbscan = DBSCAN(eps=0.25, min_samples=5)
dbscan_labels = dbscan.fit_predict(X)

# Step 4: Plot results
fig, axes = plt.subplots(1, 2, figsize=(12, 6))

# K-Means result
axes[0].scatter(X[:,0], X[:,1], c=kmeans_labels, cmap="plasma", s=50, edgecolors="k")
axes[0].set_title("K-Means (fails on circles)")

# DBSCAN result
axes[1].scatter(X[:,0], X[:,1], c=dbscan_labels, cmap="plasma", s=50, edgecolors="k")
axes[1].set_title("DBSCAN (works on circles)")

plt.show()