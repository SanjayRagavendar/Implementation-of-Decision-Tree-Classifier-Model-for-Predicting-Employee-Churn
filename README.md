# Implementation-of-Decision-Tree-Classifier-Model-for-Predicting-Employee-Churn

## AIM:
To write a program to implement the Decision Tree Classifier Model for Predicting Employee Churn.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1. Import pandas module and import the required data set.
2. Find the null values and count them.
3. Count number of left values.
4. From sklearn import LabelEncoder to convert string values to numerical values.
5. From sklearn.model_selection import train_test_split.
6. Assign the train dataset and test dataset.
7. From sklearn.tree import DecisionTreeClassifier.
8. Use criteria as entropy.
9. From sklearn import metrics.
10. Find the accuracy of our model and predict the require values.


## Program:
```py
# Developed By: Sanjay Ragavendar M K
# Register Number: 212222100045
```
```py
import pandas as pd
data=pd.read_csv('/content/Employee.csv')
data.head()
```
```py
data.info()
```
```py
data.isnull().sum()
```
```py
data["left"].value_counts()
```
```py
from sklearn.preprocessing import LabelEncoder
le= LabelEncoder()
data["salary"]=le.fit_transform(data["salary"])
data.head()
```
```py
x= data[["satisfaction_level","last_evaluation","number_project","average_montly_hours","time_spend_company","Work_accident","promotion_last_5years","salary"]]

x.head()
y=data["left"]
x
```
```py
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.2,random_state = 100)
```
```py
from sklearn.tree import DecisionTreeClassifier
dt = DecisionTreeClassifier(criterion="entropy")
dt.fit(x_train,y_train)
```
```py
y_pred = dt.predict(x_test)
from sklearn import metrics
```
```py
accuracy = metrics.accuracy_score(y_test,y_pred)
accuracy
```
```py
dt.predict([[0.5,0.8,9,260,6,0,1,2]])
```
## Output:

![Screenshot 2024-04-06 202922](https://github.com/rohithprem18/Implementation-of-Decision-Tree-Classifier-Model-for-Predicting-Employee-Churn/assets/146315115/3b6d1584-564f-4ff4-92e1-9e72a4bfd271)

![Screenshot 2024-04-06 202927](https://github.com/rohithprem18/Implementation-of-Decision-Tree-Classifier-Model-for-Predicting-Employee-Churn/assets/146315115/6fd24494-3daa-4454-b74d-4705f541bc06)

![Screenshot 2024-04-06 202931](https://github.com/rohithprem18/Implementation-of-Decision-Tree-Classifier-Model-for-Predicting-Employee-Churn/assets/146315115/0bbc04b9-2b96-477b-bad1-d910d204480e)

![Screenshot 2024-04-06 202936](https://github.com/rohithprem18/Implementation-of-Decision-Tree-Classifier-Model-for-Predicting-Employee-Churn/assets/146315115/07ba789e-e175-4018-bfd2-548037bbb6ba)

![Screenshot 2024-04-06 202941](https://github.com/rohithprem18/Implementation-of-Decision-Tree-Classifier-Model-for-Predicting-Employee-Churn/assets/146315115/fbd2b025-c2e6-42fa-9cfa-a62d95e04829)

![Screenshot 2024-04-06 202946](https://github.com/rohithprem18/Implementation-of-Decision-Tree-Classifier-Model-for-Predicting-Employee-Churn/assets/146315115/e6eebae1-b13f-4a4d-8552-2af070534275)

![Screenshot 2024-04-06 202950](https://github.com/rohithprem18/Implementation-of-Decision-Tree-Classifier-Model-for-Predicting-Employee-Churn/assets/146315115/36a73dd2-3b0e-4111-9cca-a92defe90f1d)

![Screenshot 2024-04-06 202956](https://github.com/rohithprem18/Implementation-of-Decision-Tree-Classifier-Model-for-Predicting-Employee-Churn/assets/146315115/5226bebe-5fc8-462c-b235-aafed67fc98f)



## Result:
Thus the program to implement the  Decision Tree Classifier Model for Predicting Employee Churn is written and verified using python programming.
