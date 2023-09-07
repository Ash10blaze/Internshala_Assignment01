#Customer Churn Prediction- Sunbase Data
#importing the libraries
import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
from imblearn.combine import SMOTEENN
import pickle
#importing the dataset
data=pd.read_excel('customer_churn_large_dataset.xlsx')
#viewing the dataset
#print(data.shape)
#print(data.head())
#checking info about dataset(datatypes,null values etc)
#print(data.info())
#seems like there are no null values in the dataset
#checking for duplicates in the dataset
#print(data.duplicated())
#seems like there are no duplicates in the dataset
#print(data.describe())
#dropping unwanted columns
#print(data.columns)
data.drop(columns=['CustomerID', 'Name'],inplace=True)
#checking correlations in dataset
#print(data.corr())
#selecting independent variables(input) and target variables(output)
x = data.drop('Churn',axis=1) 
y=data['Churn']
#checking unique locations
u_loc=data['Location'].unique()
#encoding these unique locations
encoded_loc=pd.get_dummies(data['Location']).astype(int)
#print(encoded_loc.head())
#dropping the original location column
x.drop('Location',axis=1,inplace=True)
#adding or concatenating
x=pd.concat([x,encoded_loc],axis=1)
#print(x)
#encoding the gender column
x['Gender']=(x['Gender']=='Male').astype(int) 
#if gender == male result will be 1 or else the result will be 0
#print(x.info())
#since all the columns are numerical lets build the model 
#Model1
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=42)
from sklearn.linear_model import LogisticRegression
reg=LogisticRegression()
reg.fit(x_train,y_train)
y_pred=reg.predict(x_test)
#checking the accuracy,f1score,precision,recall and support of the model
from sklearn.metrics import accuracy_score,classification_report,confusion_matrix
accuracy=accuracy_score(y_test,y_pred)
report=classification_report(y_test,y_pred)
c_matrix=confusion_matrix(y_test,y_pred)
#print(accuracy)
#print(report)
#print(c_matrix)
#Model2
from sklearn.ensemble import RandomForestClassifier
clf=RandomForestClassifier(n_estimators=100,random_state=42)
clf.fit(x_train,y_train)
y1_pred=clf.predict(x_test)
#print(y1_pred)
accuracy1=accuracy_score(y_test,y1_pred)
report1=classification_report(y_test,y1_pred)
c_matrix1=confusion_matrix(y_test,y1_pred)
#print(accuracy1)
#print(report1)
#print(c_matrix1)
#to improve the model1 performance we use SMOTEENN
smote_enn=SMOTEENN(random_state=42)
x_resampled,y_resampled=smote_enn.fit_resample(x,y)
xs_train,xs_test,ys_train,ys_test=train_test_split(x_resampled,y_resampled,test_size=0.2)
regsmote=LogisticRegression()
regsmote.fit(xs_train,ys_train)
y_predsmote=regsmote.predict(xs_test)
accuracy_smote=accuracy_score(ys_test,y_predsmote)
report_smote=classification_report(ys_test,y_predsmote)
c_matrix_smote=confusion_matrix(ys_test,y_predsmote)
#print(accuracy_smote)
#print(report_smote)
#print(c_matrix_smote)
#to improve the model2 performance we do the same
smote_enn=SMOTEENN(random_state=42)
x_resampled_rf,y_resampled_rf=smote_enn.fit_resample(x,y)
xs_train,xs_test,ys_train,ys_test=train_test_split(x_resampled_rf,y_resampled_rf,test_size=0.2)
clf_smote=RandomForestClassifier(n_estimators=100,random_state=42)
clf_smote.fit(xs_train,ys_train)
y_pred_rf=clf_smote.predict(xs_test)
accuracy_rf=accuracy_score(ys_test,y_pred_rf)
report_rf=classification_report(ys_test,y_pred_rf)
c_matrix_rf=confusion_matrix(ys_test,y_pred_rf)
#print(accuracy_rf)
#print(report_rf)
#print(c_matrix_rf)
#its clear that after using smoteenn the logistic regression as well as random forest classifier has given better accuracy,precision,f1score,recall etc so after using smoteenn i m choosing the random forest classifier as my final model.
with open('model.pkl', 'wb') as files:
    pickle.dump(clf_smote, files)

