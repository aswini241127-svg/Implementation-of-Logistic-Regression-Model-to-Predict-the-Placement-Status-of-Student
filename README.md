# Implementation-of-Logistic-Regression-Model-to-Predict-the-Placement-Status-of-Student

## AIM:
To write a program to implement the the Logistic Regression Model to Predict the Placement Status of Student.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1. Load and preprocess the dataset by removing unnecessary columns and converting categorical attributes into numerical values.
2. Split the dataset into training and testing sets to prepare data for model evaluation.
3.Train the Logistic Regression model using the training dataset. 
4. Train the Logistic Regression model using the training dataset.

## Program:
```
/*
Program to implement the the Logistic Regression Model to Predict the Placement Status of Student.
Developed by: ASWINI D
RegisterNumber: 25018420

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import matplotlib.pyplot as plt
import seaborn as sns


data = pd.read_csv("Placement_Data.csv")   
data = data.drop(["sl_no", "salary"], axis=1)



data["status"] = data["status"].map({"Placed": 1, "Not Placed": 0})


X = data.drop("status", axis=1)
y = data["status"]


X = pd.get_dummies(X, drop_first=True)

print("\nAfter Encoding:")
print(X.head())


scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)


X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.2, random_state=42
)


model = LogisticRegression(max_iter=1000)
model.fit(X_train, y_train)


y_pred = model.predict(X_test)
y_prob = model.predict_proba(X_test)[:, 1]


print("\nAccuracy:", accuracy_score(y_test, y_pred))

print("\nClassification Report:")
print(classification_report(y_test, y_pred))


cm = confusion_matrix(y_test, y_pred)
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("Confusion Matrix - Placement Prediction")
plt.show()


*/
```

## Output:
![Screenshot_31-1-2026_141216_127 0 0 1](https://github.com/user-attachments/assets/b60e6474-d990-4a5b-b288-45b0be819513)
![Screenshot_31-1-2026_141323_127 0 0 1](https://github.com/user-attachments/assets/2dd44b46-f886-4c5e-bc0d-c4b29a2090e8)



## Result:
Thus the program to implement the the Logistic Regression Model to Predict the Placement Status of Student is written and verified using python programming.
