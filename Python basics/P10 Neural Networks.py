# -*- coding: utf-8 -*-
"""
Created on Mon Jun  4 08:58:38 2018

@author: Abhinaba
"""
#This piece of code demonstrates the working of neural networks by solving a classification problem


#CLASSICAL NEURAL NETWORKS

#Importing the required libraries
from sklearn.neural_network import MLPClassifier #Multi-layer perceptron
import pandas as pd
import numpy as np


#reading the file
ci=pd.read_csv(r'C:\Users\Abhinaba\Desktop\Edvancer Materials\Data\census_income.csv')

#Checking few of the variable values
ci.head()

#Converting the target classification variable to 1/0 (As we would need to predict if their values are less
#than 50k or more than 50k)
ci['Y']=np.where(ci['Y']==" <=50K",0,1)



#Variable treatment
#Education
ci["education"].value_counts() #We have aroud 16 variables for education, so we may choose to delete the same

#Dropping the variable education
ci.drop('education',axis=1,inplace=True)


#We are checking out which of the variables are categorical in nature
cat_vars=ci.select_dtypes(['object']).columns

#Seeing all those variables
cat_vars


#In the steps below, we are basically creating dummified columns
for col in cat_vars:
    freqs=ci[col].value_counts() #Get all the levels and the frequency count of all the levels for a column
    k=freqs.index[freqs>100][:-1] #Get all the specific levels where frequency is greatert than 100, leaving 1 level
    for cat in k: #iterating through all the shortlisted levels
        name=col+'_'+cat #Apending the level name with the pre-defined column name
        ci[name]=(ci[col]==cat).astype(int) #Creating a new column with the name where if a shortlis level has a value, 
    del ci[col] #deleting the original column                  #then make it 1 or else make it 0 and then converting them into integer           
    print(col)
    
        

#checking the shape of the updated data frame
ci.shape #WE would get (32561,48)


#Taking a glance at the data set
ci.head()


#Splitting the data into train and test
ci_train=ci.iloc[:28000,:]
ci_test=ci.iloc[28000:,:]


#Further pre-processings
x_train=ci_train.drop('Y',axis=1)
y_train=ci_train['Y']
x_test=ci_test.drop('Y',axis=1)
y_test=ci_test['Y']



#Initialising the nn function
#(We will have 2 hidden layers, first having 20 neurons and 2nd having 10 neurons) and we are having a RELU
#function as our activation function (max(1,0))
nn=MLPClassifier(hidden_layer_sizes=(20,10),activation='relu')


#Trying to fit the data on the neural nets
nn.fit(x_train,y_train)

#We are trying to find out the predicted test score on the x_test
predicted_test_score=nn.predict_proba(x_test)[:,1]


#Checking the AUC curve
#Importing the libraries
from sklearn.metrics import  roc_auc_score

#checking the score
roc_auc_score(y_test,predicted_test_score) #We would get a score of 0.58 which is pretty bad


#Trying cross validation with multiple set of neurons in the hidden layer
#Loading the libraries required for doing cross validation
from sklearn.model_selection import KFold 


#Setting up different combos of hidden layers
hidden_layer_sizes=[(20,10),(20,11),(20,12),(20,13),(20,14)]


#Initialising an empty auc list that would capture the AUC scores of the models used across each of the
auc_scr=[]

#above mentioned hidden layers
for h in hidden_layer_sizes:
    nn=MLPClassifier(hidden_layer_sizes=h,activation='relu')
    
    kf=KFold(n_splits=10)  #WE are looking at the Kfold with number of splits=10

    xval_err=0
    
    for train, test in kf.split(x_train): 
        nn.fit(x_train.loc[train],y_train[train])
        p=nn.predict_proba(x_train.loc[test])[:,1]#We are now predicting the 10th part on the learning in 9 parts 
        auc_scr=roc_auc_score(y_train[test],p)  #Getting the auc score of each of the combinations
    
    print(h,' : ',auc_scr)    
