import pandas as pd
import numpy as np

#create a dummy dataset where feature => sixe is related to target (cancer) , 
# 0 means beneing , 1 means cancer
dataset = pd.DataFrame({"size":[4,5,6,7,25,26,27,28], 
                        "target": [0,0,0,0,1,1,1,1]})

#feature Target mapping
X = dataset[['size']]
y = dataset["target"]

#model Import - selection - fitting to dataset
from sklearn.naive_bayes import GaussianNB
model = GaussianNB()
model.fit(X,y)

##model saving
import joblib
joblib.dump(model,"Gauss_bayes_model.pkl")
print("Print Gauss_bayes_model.pkl saved")

#few details
print(f"no of classes{model.classes_}")
print(f"no of elements in each class {model.class_count_}")
print(f"no of features model trained on {model.n_features_in_}")

#little bit of visualization - KDE plots for each class
import matplotlib.pyplot as plt
import seaborn as sns

# Separate data by class
class_0_data = dataset[dataset['target'] == 0]['size']
class_1_data = dataset[dataset['target'] == 1]['size']

sns.kdeplot(class_0_data, label='Class 0 (Benign)', fill=True)
sns.kdeplot(class_1_data, label='Class 1 (Cancer)', fill=True, color='red')
plt.xlabel('Size')
plt.ylabel('Density')
plt.title('KDE Plot: Feature Distribution by Class')
plt.legend()
plt.show()



# df = pd.read_csv('german_credit_risk.csv')

# from sklearn.preprocessing import LabelEncoder
# label=LabelEncoder()

# df["Sex"]=label.fit_transform(df["Sex"])
# df["Risk"]=label.fit_transform(df["Risk"])
# df["Housing"]=label.fit_transform(df["Housing"])
# df['Purpose'] =label.fit_transform(df['Purpose'])
# df['Saving accounts'] = label.fit_transform(df['Saving accounts'])
# df['Checking account'] = label.fit_transform(df['Checking account'])

# X=df.drop(["Risk"],axis=1)
# X = X.iloc[:,1:10] #removing the index and unamed column
# y=df["Risk"]

# from sklearn.model_selection import train_test_split
# X_train, X_test, Y_train, Y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# from sklearn.naive_bayes import GaussianNB
# model = GaussianNB()
# model.fit(X_train,Y_train)

# import joblib
# joblib.dump(model,"Gauss_bayes_model.pkl")
# print("Print Gauss_bayes_model.pkl saved")

# print(f"no of classes{model.classes_}")
# print(f"no of elements in each class {model.class_count_}")
# print(f"no of features model trained on {model.n_features_in_}")

# from sklearn.metrics import confusion_matrix, classification_report
# print(confusion_matrix(Y_test,model.predict(X_test)))
# print(classification_report(Y_test,model.predict(X_test)))

# data = confusion_matrix(Y_test,model.predict(X_test))
# import seaborn as sns
# import matplotlib.pyplot as plt
# sns.heatmap(data, cmap='coolwarm')
# plt.show()