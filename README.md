# Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored

## AIM:
To write a program to predict the marks scored by a student using the simple linear regression model.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1. 
2. 
3. 
4. 

## Program:
```
/*
Program to implement the simple linear regression model for predicting the marks scored.
NAME : MOHAMMED HAMZA M 
RegisterNumber:  24900511
*/
```
import numpy as np
import matplotlib.pyplot as plt
#Preprocessing Input data
X = np.array(eval(input()))
Y = np.array(eval(input()))
#Mean
X_mean =np.mean(X)
Y_mean =np.mean(Y)
num=0 #for slope
denom=0 #for slope
#to find sum of (xi-x') & (yi-y') & (xi-x')^2
for i in range(len(X)):
    num+=(X[i] -X_mean)*(Y[i]-Y_mean)
    denom+= (X[i]-X_mean)**2
#calculate slope   
m=num/denom
#calculate intercept
b=Y_mean-m*X_mean
print(m,b)
#Line equation
y_predicted=m*X+b
print(y_predicted)
#to plot graph
plt.scatter(X,Y)
plt.plot(X,y_predicted,color='red')
plt.show()

## Output:
![Screenshot 2025-03-05 085022](https://github.com/user-attachments/assets/4b6bbdd8-2a0c-4cd5-8c1f-ebde63166609)



## Result:
Thus the program to implement the simple linear regression model for predicting the marks scored is written and verified using python programming.
