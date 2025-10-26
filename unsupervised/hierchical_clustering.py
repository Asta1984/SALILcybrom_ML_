import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import AgglomerativeClustering
from scipy.cluster.hierarchy import dendrogram, linkage

##Create A data 

X = np.array([[1,2],[1.5,1.8],[5,8],[8,8], [1,0.6], [9,11]])

##plot raw data

plt.scatter(X[:,0],X[:,1], c='black', marker='o')
plt.title("RAW DATA")
plt.show()

#step3 :: Build linkage matrix for dendogram

Z = linkage(X, method='ward') #minimize variance
plt.figure(figsize=(8,4))
dendrogram(Z)
plt.title("Dendogram hierchical clustetring")
plt.xlabel("Data Points")
plt.ylabel("Distance")
plt.show()

#apply agglomaretive clustering

cluster = AgglomerativeClustering(n_clusters=2,metric='euclidean', linkage='ward')
labels = cluster.fit_predict(X)

#plot clustered data

plt.scatter(X[:,0], X[:,0], c = labels, cmap="rainbow")
plt.title("Clusters formed by hierical clusterings")
plt.show()