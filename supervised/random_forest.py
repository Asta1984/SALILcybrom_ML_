#Importing Numpy pandas for reading data
import numpy as np
import pandas as pd

#storing data as dataframe
df = pd.read_csv('german_credit_risk.csv')

#Data preprocessing feature engineering step
df['Checking_account'] = df['Checking account'].isna().astype(int)
df['Saving_accounts'] = df['Saving accounts'].isna().astype(int)

from sklearn.preprocessing import LabelEncoder
label=LabelEncoder()

df["Sex"]=label.fit_transform(df["Sex"])
df["Risk"]=label.fit_transform(df["Risk"])
df["Housing"]=label.fit_transform(df["Housing"])
df['Purpose'] =label.fit_transform(df['Purpose'])
df['Saving accounts'] = label.fit_transform(df['Saving accounts'])
df['Checking account'] = label.fit_transform(df['Checking account'])

#Feature and target identification
X=df.drop(["Risk"],axis=1)
X = X.iloc[:,1:10]
y=df["Risk"]

#splitting the dataset
from sklearn.model_selection import train_test_split
X_train, X_test, Y_train, Y_test = train_test_split(X,y,test_size=0.3,random_state=0, stratify=y)

#The Random_Forest_classifier is imported from sklearns' ensemble methods
from sklearn.ensemble import RandomForestClassifier
model = RandomForestClassifier(n_estimators=100, criterion="entropy",random_state=42,class_weight="balanced")
model.fit(X_train,Y_train)

#model is saved to device
import joblib
joblib.dump(model, "Random_forest_ensemble.pkl")
print("Model saved sucessfully")

#Model is evaluated to check its perfomance using known metrics such as confusion matrix, classification report 
from sklearn.metrics import classification_report, confusion_matrix
print(classification_report(Y_test,model.predict(X_test)))
print(confusion_matrix(Y_test,model.predict(X_test)))

from sklearn.tree import plot_tree
import matplotlib.pyplot as plt

feature_names = ['Age', 'Sex', 'Job', 'Housing', 'Saving accounts', 'Checking account','Credit amount', 'Duration', 'Purpose']

# # Visualize the first tree
# plt.figure(figsize=(25, 15))
# plot_tree(model.estimators_[0], 
#           feature_names=feature_names,
#           class_names=['Risk', 'Good'], 
#           filled=True,
#           rounded=True)
# plt.tight_layout()
# plt.show()

from sklearn.tree import export_graphviz
import graphviz
import os
os.environ['PATH'] += os.pathsep + r'C:/Program Files/Graphviz/bin'

feature_names = ['Age', 'Sex', 'Job', 'Housing', 'Saving accounts', 'Checking account',
                 'Credit amount', 'Duration', 'Purpose']

#Export first tree to Graphviz format
dot_data = export_graphviz(model.estimators_[0],
                           feature_names=feature_names,
                           class_names=['Risk', 'Good'],
                           filled=True,
                           rounded=True,
                           out_file=None)

# Create and render the graph
graph = graphviz.Source(dot_data)
graph.render('tree', format='pdf', cleanup=True)  # Saves as tree.pdf
print("PDF saved as tree.pdf")