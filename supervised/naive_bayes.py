import pandas as pd
import numpy as np
df = pd.read_csv('german_credit_risk.csv')

from sklearn.preprocessing import LabelEncoder
label=LabelEncoder()

df["Sex"]=label.fit_transform(df["Sex"])
df["Risk"]=label.fit_transform(df["Risk"])
df["Housing"]=label.fit_transform(df["Housing"])
df['Purpose'] =label.fit_transform(df['Purpose'])
df['Saving accounts'] = label.fit_transform(df['Saving accounts'])
df['Checking account'] = label.fit_transform(df['Checking account'])

X=df.drop(["Risk"],axis=1)
X = X.iloc[:,1:10] #removing the index and unamed column
y=df["Risk"]

from sklearn.model_selection import train_test_split
X_train, X_test, Y_train, Y_test = train_test_split(X, y, test_size=0.3, random_state=42)

from sklearn.naive_bayes import GaussianNB
model = GaussianNB()
model.fit(X_train,Y_train)

import joblib
joblib.dump(model,"Gauss_bayes_model.pkl")
print("Print Gauss_bayes_model.pkl saved")

print(f"no of classes{model.classes_}")
print(f"no of elements in each class {model.class_count_}")
print(f"no of features model trained on {model.n_features_in_}")

from sklearn.metrics import confusion_matrix, classification_report
print(confusion_matrix(Y_test,model.predict(X_test)))
print(classification_report(Y_test,model.predict(X_test)))

data = confusion_matrix(Y_test,model.predict(X_test))
import seaborn as sns
import matplotlib.pyplot as plt
sns.heatmap(data, cmap='coolwarm')
plt.show()