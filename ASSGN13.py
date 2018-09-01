from sklearn.metrics import mean_absolute_error
from sklearn.metrics import mean_squared_error
from sklearn.metrics import confusion_matrix
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import seaborn as sn
import pandas as pd
import numpy as np
%matplotlib inline
#Question 1
from sklearn.datasets import load_digits
digit=load_digits()
x=digit.data
y=digit.target
plt.imshow(x[4].reshape(8,8),cmap=plt.cm.gray)

#Question 2
x_train,x_test,y_train,y_test = train_test_split(x,y,test_size = 0.3)

# Question 3
from sklearn.svm import SVC
model=SVC()
model.fit(x_train,y_train)  #kernel=gaussian
pred = model.predict(x_test)
from sklearn.metrics import classification_report
print(classification_report(y_test,pred))
from sklearn import metrics
metrics.accuracy_score(y_test,pred)
from sklearn.metrics import confusion_matrix
confusion_matrix(y_test,pred)
model1=SVC(kernel="poly") #kernel=poly
model1.fit(x_train,y_train)
pred1=model1.predict(x_test)
print(classification_report(y_test,pred1))
metrics.accuracy_score(y_test,pred1)
confusion_matrix(y_test,pred1)
model2=SVC(kernel="linear") #kernel=linear
model2.fit(x_train,y_train)
pred2=model2.predict(x_test)
print(classification_report(y_test,pred2))
metrics.accuracy_score(y_test,pred2)
confusion_matrix(y_test,pred2)
model3=SVC(kernel="sigmoid") #kernel=sigmoid
model3.fit(x_train,y_train)
pred3=model3.predict(x_test)
print(classification_report(y_test,pred3))
metrics.accuracy_score(y_test,pred3)
confusion_matrix(y_test,pred3)





