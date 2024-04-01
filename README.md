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
![image](https://github.com/SanjayRagavendar/Implementation-of-Decision-Tree-Classifier-Model-for-Predicting-Employee-Churn/assets/91368803/3f51738f-2f17-447d-bfc1-39d77cd4b552)

Displaying the head of the dataset

![image](https://github.com/SanjayRagavendar/Implementation-of-Decision-Tree-Classifier-Model-for-Predicting-Employee-Churn/assets/91368803/d3430e04-f0c7-40a6-85e6-4947ee772aa3)

Showing the information about the dataset

![image](https://github.com/SanjayRagavendar/Implementation-of-Decision-Tree-Classifier-Model-for-Predicting-Employee-Churn/assets/91368803/25173940-80b0-469d-bdc1-b70b1cf4b08b)

Printing null values in the dataset

![image](https://github.com/SanjayRagavendar/Implementation-of-Decision-Tree-Classifier-Model-for-Predicting-Employee-Churn/assets/91368803/ace51178-288a-446b-a72c-54b1d6f86646)

Value counts of the 'left' column

![image](https://github.com/SanjayRagavendar/Implementation-of-Decision-Tree-Classifier-Model-for-Predicting-Employee-Churn/assets/91368803/5bc42d5a-20e2-4615-92d4-0ea565350605)

Label encoding the values of the salary column

![image](https://github.com/SanjayRagavendar/Implementation-of-Decision-Tree-Classifier-Model-for-Predicting-Employee-Churn/assets/91368803/acdc3d89-f6ce-496c-877e-0b24533bfa19)

Spliting the columns for getting input and output

![image](https://github.com/SanjayRagavendar/Implementation-of-Decision-Tree-Classifier-Model-for-Predicting-Employee-Churn/assets/91368803/0018c2a7-1865-40fa-b475-a7ef66ab1ebf)

Creating a Decision Tree Classifier

![image](https://github.com/SanjayRagavendar/Implementation-of-Decision-Tree-Classifier-Model-for-Predicting-Employee-Churn/assets/91368803/f95b7db4-0466-4dba-8baf-2aab8da853d6)

Finding the accuracy for the test data

![image](https://github.com/SanjayRagavendar/Implementation-of-Decision-Tree-Classifier-Model-for-Predicting-Employee-Churn/assets/91368803/a3db2839-d644-4edf-b002-d0bb1b2bf0f9)

Testing the model

## Result:
Thus the program to implement the  Decision Tree Classifier Model for Predicting Employee Churn is written and verified using python programming.
