# Importing the libraries
import numpy as np
import pandas as pd

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

#Backward elimination
#import statsmodels.formula.api as sm
#X = np.append(arr = np.ones((779,1)).astype(int), values = X, axis=1)
#X_opt = X[:,[0,1,2,3,5,9,11,12,13]]
#reg_OLS = sm.OLS(endog = y, exog = X_opt).fit()
#reg_OLS.summary()

#Training set and Test set
#from sklearn.cross_validation import train_test_split
#X_train, X_test, y_train, y_test = train_test_split(X_opt, y, test_size = 0.25, random_state = 0)

#Feature Scaling
from sklearn.preprocessing import StandardScaler as ss
sc = ss()
X = sc.fit_transform(X)
X_test = sc.fit_transform(X_test)
#X_train = sc.fit_transform(X_train)
#X_test = sc.transform(X_test)

# Fitting Kernel SVM to the Training set
from sklearn.svm import SVC
classifier = SVC(kernel = 'rbf', random_state = 0)
classifier.fit(X, y)
#classifier.fit(X_train, y_train)


# Predicting the Test set results
y_pred = classifier.predict(X_test)

