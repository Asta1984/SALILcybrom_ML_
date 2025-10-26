import pandas as pd
import numpy as np

from sklearn.datasets import load_breast_cancer
feature, target = load_breast_cancer(return_X_y = True, as_frame=True) #data load
feature = feature.iloc[:,:2]

import matplotlib.pyplot as plt
plt.scatter(feature.iloc[:, 0], feature.iloc[:, 1], c=target, s=20, edgecolors="k") #plot raw data

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(feature, target, test_size=0.25) #data split

from sklearn.svm import SVC
model = SVC(kernel="linear", C=1)
model.fit(X_train, y_train)

import joblib
joblib.dump(model, "linear_svm.pkl")
print("model saved successfully")

import matplotlib.pyplot as plt #Visualize final hyperplane
from sklearn.inspection import DecisionBoundaryDisplay

DecisionBoundaryDisplay.from_estimator(model, feature, response_method="predict", cmap="coolwarm", xlabel= feature.columns[0],ylabel= feature.columns[1])

plt.scatter(feature.iloc[:, 0], feature.iloc[:, 1], c=target, s=20, edgecolors="k")
plt.show()

from sklearn.svm import SVC
model = SVC(kernel="rbf", gamma=1, C=1)
model.fit(X_train, y_train)

import joblib
joblib.dump(model, "RBF_svm.pkl")
print("model saved successfully")

import matplotlib.pyplot as plt #Visualize final hyperplane
from sklearn.inspection import DecisionBoundaryDisplay

DecisionBoundaryDisplay.from_estimator(model, feature, response_method="predict", cmap="coolwarm", xlabel= feature.columns[0],ylabel= feature.columns[1])

plt.scatter(feature.iloc[:, 0], feature.iloc[:, 1], c=target, s=20, edgecolors="k")
plt.show()