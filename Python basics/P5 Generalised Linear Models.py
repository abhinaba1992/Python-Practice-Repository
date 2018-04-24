# -*- coding: utf-8 -*-
"""
Created on Wed Apr 11 23:46:37 2018

@author: Abhinaba
"""

data_file=r'C:\Users\Abhinaba\Desktop\Edvancer Materials\Python\Data\loans data.csv'
#Setting the path from which the file shall be picked up (the commmand r tells us to read us to file path as it is)

#Importing the required libraries
import pandas as pd
import math as ma
import numpy as np
import warnings
warnings.filterwarnings('ignore')#This would help us ignore all warnings

#scikit-learn is installed using pip in cmd and included hear for implementing Linear regression, Splitting 
#of data into train and test, knowing the important values for regression etc

from sklearn.cross_validation import train_test_split #this would help to split our data into train and test
from sklearn.linear_model  import LinearRegression, Lasso, Ridge #For importing Linear Regression
from sklearn.cross_validation import KFold


ld=pd.read_csv(data_file)

ld


#Doing Basic Data Trasnformation and Feature Engineering
#Changing the data type of the 2 columns and replacing the percentage sign with blanks
for col in ["Interest.Rate","Debt.To.Income.Ratio"]:
    ld[col]=ld[col].astype("str")
    ld[col]=[x.replace("%","") for x in ld[col]]
    
    
ld.dtypes


#Seeing all the column names
list(ld)

#Convertig columns to numeric data type 
for col in [ 'Amount.Requested', 'Amount.Funded.By.Investors', 'Open.CREDIT.Lines', 'Revolving.CREDIT.Balance',
             'Inquiries.in.the.Last.6.Months', 'Interest.Rate', 'Debt.To.Income.Ratio']:
    ld[col]=pd.to_numeric(ld[col],errors="coerce") #By stating errors is equal to coerce we specify that in case
                                                   #we get a string value it will be converted to null
                                                   
#Checking the count of values in ld
ld[ 'Loan.Length'].value_counts()                                                   


#Making Loan length as it has two levels
ll_dummies=pd.get_dummies(ld["Loan.Length"]) #This will be auto converted into dummmies

ll_dummies['.'].value_counts() #We will get a count as one for the '.' column


#Subseyying the data set to see the rows that have the dot column value as 1 in the ll_dummies
ll_dummies[ll_dummies['.']==1]

#Checking the head
ll_dummies.head()

list(ll_dummies)

#Adding only one column from ll_dummies to the ld data set (by n-1 thumb rule and not including the dot column)
ld["LL_36"]=ll_dummies['36 months']

#The below code would help delete the data frame ll_dummies
%reset_selective ll_dummies

#We can also what all variables are defined in our namespace by typing who
who


#WE are now droppig the 
ld=ld.drop('Loan.Length',axis=1)


list(ld)


#Grouping variables based on the target variable when there is a large number of categories
#Checking the unique number of levels for Loan.Purpose
ld['Loan.Purpose'].nunique() #We get 14 levels

#Now grouping the categories based on the target variable
round(ld.groupby('Loan.Purpose')[ 'Interest.Rate'].mean())
#This would help us merge the categories


#Grouping the categories into one
for i in range(len(ld.index)):
    if ld['Loan.Purpose'][i] in ["car","educational","major_purchase"]:
        ld.loc[i,'Loan.Purpose']='cem'
    if ld['Loan.Purpose'][i] in ["home_improvement","medical","vacation","wedding"]:
        ld.loc[i,'Loan.Purpose']='hmvw'
    if ld['Loan.Purpose'][i] in ["credit_card","house","other","small_business"]:
        ld.loc[i,'Loan.Purpose']='chos'
    if ld['Loan.Purpose'][i] in ["debt_consolidation","moving"]:
        ld.loc[i,'Loan.Purpose']='dm'

#Seeing the new values for Loan.Purpose column        
ld['Loan.Purpose'].value_counts()                                                   
        
#We are creating dummies for this column again
lp_dummies=pd.get_dummies(ld['Loan.Purpose'],prefix="LP") #This will be auto converted into dummmies
                                                          #and the cols would be named by prefix LP

lp_dummies


list(lp_dummies)        
#Concatinating the cols of lp_dummies with the data frame ld (note that since we know that the number 
#of rows for both of the data frames are same and that the index value is also same, we are concatinating
#it, in case we do not know if they are same, we do a left join)
ld=pd.concat([ld,lp_dummies],1)

#Dropping the unnecessary columns
ld=ld.drop(['Loan.Purpose','LP_renewable_energy'],1)

#deleting the dummy data set
%reset_selective lp_dummies


#State variable
ld['State'].nunique()


#Dropping the state column since it has 47 levels
ld=ld.drop(['State'],1) 


#Home ownership
ld['Home.Ownership'].value_counts()

#Creating dummies
ld['ho_mort']=np.where(ld['Home.Ownership']=='MORTGAGE',1,0)
ld['ho_rent']=np.where(ld['Home.Ownership']=='RENT',1,0)
ld=ld.drop(['Home.Ownership'],1)


#Fico.range
ld['FICO.Range'].head()

#We are now trying to seperate out the two values seperated by a '-' and convert them into numeric
ld['f1'],ld['f2']=zip(*ld['FICO.Range'].apply(lambda x: x.split('-')))
#Zip here would help me get the result returned in tuple format


#We are now taking an average of the two variables and deleting the two dummies
ld['fico']=0.5*(pd.to_numeric(ld['f1'])+pd.to_numeric(ld['f2']))
ld=ld.drop(['FICO.Range','f1','f2'],1) 


#Employment length
ld['Employment.Length'].value_counts()


#Refining the variable employment length
ld['Employment.Length']=ld['Employment.Length'].astype('str')
ld['Employment.Length']=[x.replace('years','') for x in ld['Employment.Length']]
ld['Employment.Length']=[x.replace('year','') for x in ld['Employment.Length']]

#Checking the years variable again
ld['Employment.Length'].value_counts()


#Further conversions
ld['Employment.Length']=[x.replace('n/a','<1') for x in ld['Employment.Length']]
ld['Employment.Length']=[x.replace('10+','10') for x in ld['Employment.Length']]
ld['Employment.Length']=[x.replace('<1','0') for x in ld['Employment.Length']]
ld['Employment.Length']=pd.to_numeric(ld['Employment.Length'],errors='coerce')

#Checking the variables
ld['Employment.Length'].value_counts()

#So all the employment length variables are now converted to numeric

#Dropping the NA's
ld.dropna(axis=0,inplace=True)


#We are now ready to do the modelling on our data since our data prep is ready

#Splitting the data into train and test
ld_train, ld_test = train_test_split(ld,test_size=0.2,random_state=2)

#Initiating the linear regression object
lm=LinearRegression()


#Dropping the unnecessary columns from train and test data set and seperating out the independent 
#variables and dependent variables
x_train=ld_train.drop(['Interest.Rate','ID','Amount.Funded.By.Investors'],1)
y_train=ld_train['Interest.Rate']
x_test=ld_test.drop(['Interest.Rate','ID','Amount.Funded.By.Investors'],1)
y_test=ld_test['Interest.Rate']



#Invoking the fit function of the lm object so as to run the model
lm.fit(x_train,y_train)
#So the below sentence would actually run when we execute the above line of code
#LinearRegression(copy_X=True, fit_intercept=True, n_jobs=1, normalize=False)
#n_jobs would parallalize the modelling process and run it a bit faster
#normalize helps us to normalize the data for modelling
#fit_intercept=true is important and it helps us to give a base effect to out model,
#in y=mx+c, its the c or beta 0 in a LM equation

#We can check the coefficients in our model by running the following
lm.coef_ #We will get an array of coeffcients


#Prediction and errorz
#trying to fit our model on test data
p_test=lm.predict(x_test) #Predicted value on test data

residual=p_test-y_test  #Errors or residual on test data

#Checking the RMSE value
rmse_lm=np.sqrt(np.dot(residual,residual)/len(p_test))

rmse_lm #We get an RMSE value of 2.16 which is pretty good


#Getting all the coefficients
coefs=lm.coef_
coefs

#Getting all the features 
features=x_train.columns
features

#If we wish to get the features and the columns in a single set together, we do the following
list(zip(features,coefs))


#Getting the intercept of the model
lm.intercept_ #We will get 75.88


#Regularised form of regression

#First we need to find the best value for penalty weight with cross validation for ridge regression
alphas=np.linspace(.0001,10,100)
#We are resetting the index for cross validation to work without hitch
x_train.reset_index(drop=True,inplace=True)
y_train.reset_index(drop=True,inplace=True)



#We are using MSE instead of RMSE to validate our model for ridge regression
#Importing the below for the same
from sklearn.metrics import mean_squared_error

rmse_list=[]
for a in alphas:
    ridge=Ridge(fit_intercept=True, alpha=a)
    
    #Calculating average RMSE across 10-fold cross validation
    kf=KFold(len(x_train),n_folds=10)
    xval_err=0
    for train, test in kf:
        ridge.fit(x_train.loc[train],y_train[train])
        p=ridge.predict(x_train.loc[test])
        #error = p - y_train[test]
        #xval_err += np.dot(err,err)
        xval_err += mean_squared_error()
    
    #rmse_10cv=np.sqrt(xval_err/len(x_train))
    rmse_10cv=xval_err/10
    #Uncomment below to print rmse values for individual alphas
    #print('{:.3f}\t {:.6f}\t '.format(a,rmse_10cv))
    rmse_list.extend([rmse_10cv])

best_alpha=alphas[rmse_list==min(rmse_list)]
print('Alpha with min 10cv error is: ',best_alpha)
