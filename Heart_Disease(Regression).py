# Importing the libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Importing the dataset(Train)
dataset = pd.read_csv('Data_Train.csv')
X = dataset.iloc[:, :-1].values
y = dataset.iloc[:, 13].values

# Importing the dataset(Test)
dataset = pd.read_csv('Data_Test.csv')
X_test = dataset.iloc[:, :].values

# Missing data
from sklearn.preprocessing import Imputer
imputer = Imputer(missing_values = 'NaN', strategy = 'mean', axis = 0)
imputer = imputer.fit(X[:, 0:13])
X[:, 0:13] = imputer.transform(X[:, 0:13])
#TEST
imputer_test = imputer.fit(X_test[:,:])
X_test[:,:] = imputer_test.transform(X_test[:,:])

#Splitting the X into X_tr and X_te 
from sklearn.model_selection import train_test_split as tts
X_tr,X_te,y_tr,y_te = tts(X , y , test_size = 0.20 , random_state = 0)

#Feature scaling
from sklearn.preprocessing import StandardScaler as ss
sc = ss()
X = sc.fit_transform(X)
X_tr = sc.fit_transform(X_tr)
X_te = sc.fit_transform(X_te)
X_test = sc.fit_transform(X_test)

#Fitting linear regression
from sklearn.svm import SVR
regressor = SVR(kernel = 'rbf')
regressor.fit(X_tr,y_tr)
regressor_1 = SVR(kernel = 'rbf')
regressor_1.fit(X,y)

#Predicting the values
    #Predicting the values of traing set
y_pred_train = regressor.predict(X_tr)
y_pred_train = np.round(y_pred_train)
     #Predicting the values of test set
y_pred_test = regressor.predict(X_te)
y_pred_test = np.round(y_pred_test)

#Confusion matrix
from sklearn.metrics import confusion_matrix
#Train
cm_train = confusion_matrix(y_tr,y_pred_train)
#Test
cm_test = confusion_matrix(y_te,y_pred_test)

print("The Training data set is split into train set and test set (2:8)")

print("Accuracy of train set is :",((cm_train[0][0]+cm_train[1][1])/623)*100 , "%" )

print("Accuracy of test set is  : " , ((cm_test[0][0]+cm_test[1][1])/156)*100 , " %")

#Predicting the test values
y_pred = regressor_1.predict(X_test)
y_pred = np.round(y_pred)
