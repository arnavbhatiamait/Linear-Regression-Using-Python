# Simple Linear Regression
# Importing the Libraries
# 
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error 
import sklearn as sk

# Importing the Data Set 

df=pd.read_csv("Salary_Data.csv")
df

x=df.iloc[:,:-1].values
# x
y=df.iloc[:,-1].values
# y

# Splitting the Data Set In testing and Training Set 

x_test,x_train,y_test,y_train=train_test_split(x,y,test_size=0.8,random_state=0 )
# x_train
# x_test
# y_test
# y_test

# Training the Simple Linear Regression Model

linearReg=sk.linear_model.LinearRegression()
linearReg.fit(x_train,y_train)
# print(y_pred)

# Predicting the Test set Results 


y_pred=linearReg.predict(x_test)
y_linear=linearReg.predict(x_train)

# y_pred

# Visualizing the Training set Results

plt.scatter(x_train,y_train,color='red')
plt.plot(x_train,y_linear,color='blue')
plt.title("Salary Vs Experience")
plt.xlabel("Years of Experience")
plt.ylabel("Salary")
plt.show()

# Visualizing the Test set Results

plt.scatter(x_test,y_test,color='red')
# plt.plot(x_test,y_pred,color='blue')
plt.plot(x_train,y_linear,color='blue')
# ! ypred and ylinear will be same
plt.title("Salary Vs Experience")
plt.xlabel("Years of Experience")
plt.ylabel("Salary")
plt.show()

# Mean Squared Error
MSE = np.square(np.subtract(y_test,y_pred)).mean() 
print(MSE)
print(mean_squared_error(y_test,y_pred))

















































































































# for i in range (6):
#     print(y_pred[i],"and",y_test[i], 'and',y_test[i]-y_pred[i])
# print(y_test)

# Confussion matrix
# from sklearn import metrics
# import seaborn as sns
# from sklearn.metrics import confusion_matrix  
# cm= confusion_matrix(y_test, y_pred) 
# cm
# from sklearn.metrics import classification_report
# print(classification_report(y_test, y_pred))