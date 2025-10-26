import numpy as np
from sklearn.svm import SVC
import matplotlib.pyplot as plt

#Create a dummy dataset
X = np.array([[2,2],[4,4],[2,0],[0,2]])
y = np.array([1,1,-1,-1])

#train a linearn svm model increse C to increse Margin

model = SVC(kernel="linear", C=1e6)
model.fit(X,y)

w = model.coef_[0]
b = model.intercept_[0]
support_vectors = model.support_vectors_
support_idx = model.support_

print("w (coefficients):", w)
print("b (intercept):", b)
print("support vectors \n", support_vectors)
print("Support indices (in original X):", support_idx)

#Compute margin distances
w_norm = np.linalg.norm(w)
dist_to_margin = 1.0/w_norm
full_margin = 2.0/w_norm
print("||w||=",w_norm)
print("Dist from boundary to supoort hyperplane:", dist_to_margin)
print("full margin:", full_margin)

#4) plot data visualize
plt.figure(figsize=(6,6))

#plot point

plt.scatter(X[y==1,0], X[y==1,1], marker="o", label="class +1")
plt.scatter(X[y==-1,0], X[y==-1,1], marker="s", label="class -1")

#highlight the support vector
plt.scatter(support_vectors[:,0], support_vectors[:,1], s=200, facecolors='none', edgecolors='k', linewidths=2, label="support vectors")

#Decision boundary: W1*x + w2*y + b = 0 
xx = np.linspace(-1,6,200)
yy_decision = -(w[0]*xx + b) / w[1]
yy_margin_pos = -(w[0]*xx + b - 1) / w[1]
yy_margin_neg = -(w[0]*xx + b + 1) / w[1]

plt.plot(xx, yy_decision, linestyle = "-", linewidth = 2, label = "decision boundary")
plt.plot(xx, yy_margin_pos, linestyle = "--", linewidth = 1, label = "margin (+1)")
plt.plot(xx, yy_margin_neg, linestyle = "--", linewidth = 1, label = "margin (-1)")

plt.xlim(-1,6)
plt.ylim(-1,6)
plt.xlabel('x1')
plt.ylabel('x1')
plt.legend()
plt.title('Linear SVM - Decision boundary and margin')
plt.gca().set_aspect('equal', adjustable='box')
plt.show()



#Example the trained model is used to predict

new_points = np.array([[4,4],[2,0],[3,1],[1,1]])
predictions = model.predict(new_points)
print("predictions:",predictions)
scores = model.decision_function(new_points)
print("Decision function values:", scores)