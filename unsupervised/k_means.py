#kmeans from scartch 
import numpy as np
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt

points = np.array([[1,2],[1,4],[1,0],[5,2],[5,4],[6,2]])

labels = ["A","B","C","D","E","F"]

#step 2 run Kmeans with 2 cluster

kmeans = KMeans(n_clusters=2,random_state=42, n_init=10, random_state=42)
kmeans.fit(points)
clusters = kmeans.labels_
centroids = kmeans.cluster_centers_

#steps 3 Print results
print("clusters Assignments")
for i, labels in enumerate(labels):
    print(f"Points {labels} {points[i]} -> Cluster {clusters[i]}")

print('\nFinal Centroids:')
print(centroids)

#Visualizations 
plt.figure(figsize=(6,6))
plt.scatter(points[:,0],points[:,1],c = clusters, s=100, cmap="viridis", label="Points")

#Mark centroids 
plt.scatter(centroids[:,0],centroids[:,1],c="red", marker="X", s= 200, label = "centroids")

#Annotate the labels:
for i, txt in enumerate(labels):
    plt.annotate(txt, (points[i,0]+0.1, points[i,1]+0.1))


plt.title("Kmeans Clustering Example")
plt.xlabel("X1")
plt.ylabel("X2")
plt.legend()
plt.grid(True)
plt.show()