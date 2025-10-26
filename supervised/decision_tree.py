#import basic Libraries
import numpy as np
import pandas as pd

#Import dataset
df = pd.read_csv('german_credit_risk.csv')

#perform data preprocessing convert categorical data to numerical data
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

# Set Features and Targets as we require them in supervised learning
X=df.drop(["Risk"],axis=1)
X = X.iloc[:,1:10]
y=df["Risk"]

#Split the dataset into trainning and testing sets for cross validation 
from sklearn.model_selection import train_test_split
X_train, X_test, Y_train, Y_test = train_test_split(X,y,test_size=0.2,random_state=42, stratify=y)

#import Required model and fit it to training dataset
from sklearn.tree import DecisionTreeClassifier
from sklearn import tree

from sklearn.tree import DecisionTreeClassifier
model = DecisionTreeClassifier(criterion='entropy',splitter="best", max_depth=5,min_samples_split=20,min_samples_leaf=10,random_state=0,class_weight='balanced')
model.fit(X_train,Y_train)

#save the model using joblib
import joblib
joblib.dump(model, "decision_tree.pkl")
print("Model saved sucessfully")

#Evaluate models for their accuracy, precision, F1-score and other metrics 
from sklearn.metrics import classification_report, accuracy_score
print(classification_report(Y_test,model.predict(X_test)))


from sklearn import tree
tree.plot_tree(model)
