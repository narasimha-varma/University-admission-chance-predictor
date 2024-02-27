import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
#load the data into pandas dataframe
data=pd.read_csv('admission.csv')
# print(data.describe().to_string())
X=data[['GRE Score','TOEFL Score','University Rating','SOP','LOR ','CGPA','Research']]
Y=data['Chance of Admit ']
#convert the data into array for calculations
x=np.array(X)
y=np.array(Y)
#split the data into 8:2 ratio
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=0)
print(x_train.shape,y_test.shape)
#train the model using linear regression
from sklearn.linear_model import LinearRegression
reg=LinearRegression().fit(x_train,y_train)
#Accuracy of the model
print(reg.score(x_train,y_train))
y_train_predict=reg.predict(x_train)
y_predict=reg.predict(x_test)
print(pd.DataFrame({'y_train':y_train,'y_train_predict':y_train_predict}))
print(pd.DataFrame({'y_test':y_test,'y_predict':y_predict}))
#accuracy on train data
from sklearn import metrics
print(metrics.mean_absolute_error(y_train,y_train_predict))
print(metrics.mean_squared_error(y_train,y_train_predict))
print(np.sqrt(metrics.mean_squared_error(y_train,y_train_predict)))
#accuracy of train set and test set
print(metrics.r2_score(y_test, y_predict))
print(metrics.r2_score(y_train,y_train_predict))
print(reg.score(x,y))


