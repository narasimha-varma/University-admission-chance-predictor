import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
data=pd.read_excel('admission.xls',engine='xlrd')
print(data.describe())
print(data)
X1=data['GRE Score']
Y=data['Chance of Admit ']
x1=np.array(X1)
y=np.array(Y)
plt.plot(x1,y,'o')
plt.show()
x1=x1.reshape(-1,1)
reg=LinearRegression().fit(x1,y)
print(reg.score(x1,y))
def line(x1):
    m=reg.coef_
    c=reg.intercept_
    Y=m*x1+c 
    plt.plot(x1,y,'o')
    plt.plot(x1,Y)
line(x1)    
