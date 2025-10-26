import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_circles
from sklearn.svm import SVC

#create a custom dataset
X, y = make_circles(n_samples=200, factor=0.5, noise=0.05)
#convert 0/1 labels to -1/+1 for clarity
y = np.where(y == 0, -1, 1)


#Train SVM with RBF kernel
model = SVC(kernel="rbf", gamma=1.0, C=1.0)
model.fit(X,y)

#plot a decision boundary
plt.figure(figsize=(6,6))
#create a meshgrid for visualization
xx, yy = np.meshgrid(np.linspace(-1.5, 1.5, 300), np.linspace(-1.5, 1.5, 300))
Z = model.decision_function(np.c_[xx.ravel(),yy.ravel()])

Z = Z.reshape(xx.shape)

#plot decision boundary and margins
plt.contourf(xx, yy, Z>0, alpha=0.2)
plt.contour(xx, yy, Z, levels=[-1, 0, 1], linestyles=["--", "-","--"], colors='k')

#plot the data points

plt.scatter(X[y==1, 0],X[y==1, 1], color='blue', label="class +1")
plt.scatter(X[y==-1, 0],X[y==-1, 1], color='red', label="class -1")
plt.legend()
plt.show()