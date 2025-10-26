import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

fruit_data = pd.read_csv('fruit_data_with_colors.csv')

# #raw data visualization 
X_feature = 'width'
Y_feature = 'height'

# Define colors for the 4 unique fruit types
fruit_names = fruit_data['fruit_name'].unique()
colors = ['red', 'green', 'blue', 'orange']
# Map each fruit name to a specific color
color_map = dict(zip(fruit_names, colors[:len(fruit_names)]))
# The mapping is:
# 'apple': 'red'
# 'mandarin': 'green'
# 'orange': 'blue'
# 'lemon': 'orange' 

# --- Plot Generation ---
plt.figure(figsize=(10, 8))

# Iterate over each unique fruit and plot its data points
for fruit_name in fruit_names:
    subset = fruit_data[fruit_data['fruit_name'] == fruit_name]
    plt.scatter(
        subset[X_feature],
        subset[Y_feature],
        c=color_map[fruit_name], # Use the assigned color
        label=fruit_name,       # Use fruit name for the legend
        alpha=0.8,              # Set transparency
        s=100                   # Set marker size
    )

# --- Plot Customization ---
plt.xlabel(X_feature.capitalize(), fontsize=14)
plt.ylabel(Y_feature.capitalize(), fontsize=14)
plt.title('Fruit Classification Scatter Plot (Width vs. Height)', fontsize=16)
plt.legend(title='Fruit Type', loc='best')
plt.grid(True, linestyle='--', alpha=0.6)

plt.show()

#Feature Engineering feature identification and setting them into X,y
X=fruit_data[['width','height']]
y=fruit_data['fruit_label']

#creating a trainning and testing set
from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test = train_test_split(X,y, test_size=0.22)

#import KNeighborsClassifier model and fit the model to dataset
from sklearn.neighbors import KNeighborsClassifier
model = KNeighborsClassifier(n_neighbors=3)
model.fit(X_train,y_train)

#save the model using joblib
import joblib
joblib.dump(model, "KNN_fruit_class.pkl")
print("Model saved sucessfully")

#Scoring a model means measuring it efficacy (how good a model is performing?)
from sklearn.metrics import confusion_matrix, classification_report
print(confusion_matrix(y_test, model.predict(X_test)))
print(model.score(X_test,y_test))


#classifying unknown data point (konsa fruit hoga given uska width, height)
#ek chota sa helper function

def helper(x):
    if x == 1:
        print("Fruit Detected: APPLE") 
    elif x == 2:
        print("Fruit Detected: Mandarin")
    elif x==3:
        print("Fruit Detected: Orange")
    else: 
        print("Fruit Detected: lemon ") 


Unknown_fruit = model.predict(np.array([[6.3, 8]]))

print(helper(Unknown_fruit))