import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# ##Create a dataset of random number
# data = np.random.randint(1,10,100)
# data = data.reshape(50,2)

# data = pd.DataFrame(data)

# X = data.iloc[:, 0]
# Y = data.iloc[:, 1]

# # X = data[0]
# # y = data[1]

# plt.scatter(X, Y)
# plt.show()

# m = 0
# c = 0

# L = 0.01  # The learning Rate
# epochs = 10000  # The number of iterations to perform gradient descent

# n = float(len(X)) # Number of elements in X

# # Performing Gradient Descent

# for i in range(epochs):
#     Y_pred = m*X + c 

#     D_m = (-2/n) * sum(X * (Y - Y_pred))  # Derivative wrt m
#     D_c = (-2/n) * sum(Y - Y_pred)  # Derivative wrt c

#     m = m - L * D_m  # Update m
#     c = c - L * D_c  # Update c

# print (m, c)



#Using the inbuilt scikit_learn method to calculate Gradiant descent

from sklearn.datasets import load_diabetes
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
from sklearn.model_selection import train_test_split

## spliting dataset into train-test set
X,y = load_diabetes(return_X_y=True)
X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.2,random_state=2)


##Fitting the model
model = LinearRegression()
model.fit(X_train,y_train)


##Measuring model losses using 
y_pred = model.predict(X_test)
r2_score(y_test,y_pred)



#SGD Gradient descent method of scikit-learn
from sklearn.linear_model import SGDRegressor

reg = SGDRegressor(max_iter=10000,learning_rate='constant',eta0=0.01)
reg.fit(X_train,y_train)

y_pred = reg.predict(X_test)
print(r2_score(y_test,y_pred))
print(reg.coef_) 
print(reg.intercept_)