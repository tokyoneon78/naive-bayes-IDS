#Importing necessary Libraries
import pandas as pd
from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from sklearn.metrics import accuracy_score, confusion_matrix
import numpy as np

#loading the dataset 
df = pd.read_csv('./NSL.csv')

#Preprocessing the dataset 
cols = list(df.columns)
cols.remove('Class')
df.dropna(inplace=True)
df_min_max_scaled = df.copy()
columns = ['A', 'C', 'D']
for column in columns:
    columns = ['A', 'C', 'D']
for column in columns:
  df_min_max_scaled[column] = (df_min_max_scaled[column] - df_min_max_scaled[column].min()) / (df_min_max_scaled[column].max() - df_min_max_scaled[column].min())
df_min_max_scaled['AO'].apply(lambda x:x+1)
for col in cols:
  df_min_max_scaled[col] = np.log(df_min_max_scaled[col].apply(lambda x:x+1))

#Plot processed data
df_min_max_scaled.hist(figsize=(15,20))

#print dataset shape
X = df_min_max_scaled.drop('Class', axis=1)
Y = df_min_max_scaled['Class']
print("DataSet Size: ", X.shape, Y.shape)

#Spiliting the dataset into train and test
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, train_size=0.80, test_size=0.20, stratify=Y, random_state=123)
print('Train/Test Sizes : ', X_train.shape, X_test.shape, Y_train.shape, Y_test.shape)

#Fitting the data into the model
gaussian_nb = GaussianNB()
gaussian_nb.fit(X_train, Y_train)

predict_train = gaussian_nb.fit(X_train, Y_train).predict(X_train)
print("Accuracy of the model based on training data is {score}", accuracy_score(Y_train, predict_train))

predict_test = gaussian_nb.fit(X_test, Y_test).predict(X_test)
print("Accuracy of the model based on testing data is {score}", accuracy_score(Y_test, predict_test))

print("Cross Validation scores are {score}", cross_val_score(gaussian_nb, X_train, Y_train, cv=5))


