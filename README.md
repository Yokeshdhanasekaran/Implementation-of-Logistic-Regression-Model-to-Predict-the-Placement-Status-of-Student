# Implementation-of-Logistic-Regression-Model-to-Predict-the-Placement-Status-of-Student

## AIM:
To write a program to implement the the Logistic Regression Model to Predict the Placement Status of Student.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1.Import the standard libraries.

2.Upload the dataset and check for any null or duplicated values using .isnull() and .duplicated() function respectively.

3.Import LabelEncoder and encode the dataset.

4.Import LogisticRegression from sklearn and apply the model on the dataset.

5.Predict the values of array.

6.Calculate the accuracy, confusion and classification report by importing the required modules from sklearn.

7.Apply new unknown values

## Program:

```
Program to implement the the Logistic Regression Model to Predict the Placement Status of Student.
Developed by: YOKESH.D
RegisterNumber:  2122222220061

import pandas as pd
data=pd.read_csv('/content/Placement_Data.csv')
data.head()

data1=data.copy()
data1=data1.drop(["sl_no","salary"],axis=1)
data1.head()

data1.isnull().sum()

data1.duplicated().sum()

from sklearn.preprocessing import LabelEncoder
le=LabelEncoder()
data1["gender"] = le.fit_transform(data1["gender"])
data1["ssc_b"] = le.fit_transform(data1["ssc_b"])
data1["hsc_b"] = le.fit_transform(data1["hsc_b"])
data1["hsc_s"] = le.fit_transform(data1["hsc_s"])
data1["degree_t"] = le.fit_transform(data1["degree_t"])
data1["workex"] = le.fit_transform(data1["workex"])
data1["specialisation"] = le.fit_transform(data1["specialisation"])
data1["status"] = le.fit_transform(data1["status"])
data1

x=data1.iloc[:,:-1]
x

y=data1["status"]
y

from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test = train_test_split(x,y,test_size = 0.2,random_state = 0)

from sklearn.linear_model import LogisticRegression
lr=LogisticRegression(solver = "liblinear")
lr.fit(x_train,y_train)
y_pred=lr.predict(x_test)
y_pred


from sklearn.metrics import accuracy_score
accuracy=accuracy_score(y_test,y_pred)
accuracy

from sklearn.metrics import confusion_matrix
confusion=(y_test,y_pred)
confusion

from sklearn.metrics import classification_report
classification_report1=classification_report(y_test,y_pred)
print(classification_report1)

lr.predict([[1,80,1,90,1,1,90,1,0,85,1,85]])



```

## Output:
### placement data:
![image](https://github.com/ashwinkumarsaveethaofficial/Implementation-of-Logistic-Regression-Model-to-Predict-the-Placement-Status-of-Student/assets/120731469/f9c67794-434a-45f2-80e8-543387d0676b)

### salary data:
![image](https://github.com/ashwinkumarsaveethaofficial/Implementation-of-Logistic-Regression-Model-to-Predict-the-Placement-Status-of-Student/assets/120731469/2b33480e-44d9-4716-84e9-cb4dab3f6415)

### checking the null function:
![image](https://github.com/ashwinkumarsaveethaofficial/Implementation-of-Logistic-Regression-Model-to-Predict-the-Placement-Status-of-Student/assets/120731469/7a4ca9bf-f7c2-49be-8714-7498534b527d)

### data duplicate:
![270336101-d89cf75f-e006-4e92-b827-9e7442f8523e](https://github.com/ashwinkumarsaveethaofficial/Implementation-of-Logistic-Regression-Model-to-Predict-the-Placement-Status-of-Student/assets/120731469/e5b2de13-8f72-4663-b678-741213fe42e9)

### print data :
![image](https://github.com/ashwinkumarsaveethaofficial/Implementation-of-Logistic-Regression-Model-to-Predict-the-Placement-Status-of-Student/assets/120731469/ff537fab-1fc0-4c1e-8461-1fb0f144f4d8)

### data status:
![image](https://github.com/ashwinkumarsaveethaofficial/Implementation-of-Logistic-Regression-Model-to-Predict-the-Placement-Status-of-Student/assets/120731469/a060d69e-9439-4280-9982-1806196121a1)

### y_prediction array:
![image](https://github.com/ashwinkumarsaveethaofficial/Implementation-of-Logistic-Regression-Model-to-Predict-the-Placement-Status-of-Student/assets/120731469/d8e76d98-c256-4a38-8270-d303664f5e04)

### accuracy value:
![image](https://github.com/ashwinkumarsaveethaofficial/Implementation-of-Logistic-Regression-Model-to-Predict-the-Placement-Status-of-Student/assets/120731469/96d58c3c-dcde-4e6c-83e2-f3d06f5ed604)

### confusion array:
![image](https://github.com/ashwinkumarsaveethaofficial/Implementation-of-Logistic-Regression-Model-to-Predict-the-Placement-Status-of-Student/assets/120731469/01881216-34c3-494e-b278-a299e972d8f1)

### classification report:
![image](https://github.com/ashwinkumarsaveethaofficial/Implementation-of-Logistic-Regression-Model-to-Predict-the-Placement-Status-of-Student/assets/120731469/0374c88b-80d6-4a7b-9c25-eeb630938ee7)

### prediction of LR:
![image](https://github.com/ashwinkumarsaveethaofficial/Implementation-of-Logistic-Regression-Model-to-Predict-the-Placement-Status-of-Student/assets/120731469/fd24f9d9-d6f4-4c40-9756-88abf175643b)






## Result:
Thus the program to implement the the Logistic Regression Model to Predict the Placement Status of Student is written and verified using python programming.
