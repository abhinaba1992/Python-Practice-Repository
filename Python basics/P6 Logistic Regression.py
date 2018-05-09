# -*- coding: utf-8 -*-
"""
Created on Fri May  4 10:24:02 2018

@author: Abhinaba
"""
#Importing the required libraries
import pandas as pd
import math as ma
import numpy as np
import warnings
warnings.filterwarnings('ignore')#This would help us ignore all warnings

#Importing logistic regression specific components
from sklearn.cross_validation import train_test_split #this would help to split our data into train and test
from sklearn.linear_model import LogisticRegression #For the logistic regression
from sklearn.metrics import roc_auc_score #For checking the AUC score


data_file=r'C:\Users\Abhinaba\Desktop\Edvancer Materials\Python\Data\Existing Base.csv'
#Setting the path from which the file shall be picked up (the commmand r tells us to read us to file path as it is)

#Reading the required file
bd=pd.read_csv(data_file)

bd.head()

#Checking the row and column counts for the data set
bd.shape


#Variable treatment and data preparation
#Children
bd["children"].value_counts() #Checking the count of different categories of children

#Doing trasnformation on the column children to make it into a numeric col
bd.loc[bd["children"]=="Zero","children"]="0"
bd.loc[bd["children"]=="4+","children"]="4"
bd["children"]=pd.to_numeric(bd["children"],errors="coerce")


#Revenue Grid
bd["Revenue Grid"].value_counts()

#Creating a new variable y based on the values of the "Revenue Grid" column
bd["y"]=np.where(bd["Revenue Grid"]==2,0,1)
#Dropping the main column ["Revenue Grid"] as we will not need it now
bd=bd.drop(["Revenue Grid"],1)

#Checking the columns
bd.columns

bd["y"].value_counts()


#Age bands
bd["age_band"].value_counts()

#Checking the mean sensitivity towards y column
bd.groupby("age_band")["y"].mean().round(2) #This is the equivalent of tapply in R


#Doing transformation for the age_band column
for i in range(len(bd)):
    if bd["age_band"][i] in ["71+","65-70","51-55","45-50"]:
        bd.loc[i,"age_band"]="ab_10"
    if bd["age_band"][i] in ["55-60","41-45","31-35","22-25","26-30"]:
        bd.loc[i,"age_band"]="ab_11"
    if bd["age_band"][i] in ["36-40"]:
        bd.loc[i,"age_band"]="ab_13"
    if bd["age_band"][i] in ["18-21"]:
        bd.loc[i,"age_band"]="ab_17"
    if bd["age_band"][i] in ["61-65"]:
        bd.loc[i,"age_band"]="ab_9"
        
#Creating dummies for that column
ab_dummies=pd.get_dummies(bd["age_band"])
ab_dummies.head()
        
    
        
#Adding the age band dummies to the main data set
bd=pd.concat([bd,ab_dummies],1)

#Dropping the cols that wont be required anymore
bd=bd.drop(["age_band","Unknown"],1) #age_band is dropped coz it won't be required, Unknown is dropped because
                                     #of the n-1 rule
                                     
#Status                                     
bd["status"].value_counts()


#Creating the dummies for the same
bd["st_partner"]=np.where(bd["status"]=="Partner",1,0)
bd["st_singleNm"]=np.where(bd["status"]=="Single/Never Married",1,0)
bd["st_divSep"]=np.where(bd["status"]=="Divorced/Separated",1,0)


#Dropping the columns that aren't required now
bd=bd.drop(["status"],1)


#Occupation
bd["occupation"].value_counts()


#Checking the mean sensitivity towards y column
bd.groupby("occupation")["y"].mean().round(2) #This is the equivalent of tapply in R


#Doing transformation for the occupation column
for i in range(len(bd)):
    if bd["occupation"][i] in ["Unknown","Student","Secretarial/Admin","Other","Manual Worker"]:
        bd.loc[i,"occupation"]="oc_11"
    if bd["occupation"][i] in ["Professional","Business Manager"]:
        bd.loc[i,"occupation"]="oc_12"
    if bd["occupation"][i] in ["Retired"]:
        bd.loc[i,"occupation"]="oc_10"

oc_dummies=pd.get_dummies(bd["occupation"])
oc_dummies.head()


bd=pd.concat([bd,oc_dummies],1)

bd=bd.drop(["occupation","Housewife"],1)

#occupation partner
bd["occupation_partner"].value_counts()

#Checking the mean sensitivity towards y column
bd.groupby("occupation_partner")["y"].mean().round(2) #This is the equivalent of tapply in R

#Doing transformation for the occupation column
bd["ocp_10"]=0
bd["ocp_12"]=0

for i in range(len(bd)):
    if bd["occupation_partner"][i] in ["Unknown","Retired","Other"]:
        bd.loc[i,"ocp_10"]=1
    if bd["occupation_partner"][i] in ["Student","Secretarial/Admin"]:
        bd.loc[i,"ocp_12"]=1
    
    
    
#Dropping all the unecessary columns now
bd=bd.drop(["occupation_partner","TVarea","post_code","post_area","region"],1)
        
        
#Home status
bd["home_status"].value_counts()


#Doing the trasnformation
bd["hs_own"]=np.where(bd["home_status"]=="Own Home",1,0)
bd["hs_council"]=np.where(bd["home_status"]=="Rent from Council/HA",1,0)
bd["hs_RentPri"]=np.where(bd["home_status"]=="Rent Privately",1,0)
bd["hs_ParentHom"]=np.where(bd["home_status"]=="Live in Parental Hom",1,0)

del bd["home_status"] #Another syntax to drop a col from a data set


#Gender
bd["gender"].value_counts()

#Doing the trasnformation
bd["gender_f"]=np.where(bd["gender"]=="Female",1,0)

del bd["gender"]


#Self employed
bd["self_employed"].value_counts()


#Doing the trasnformation
bd["semp_yes"]=np.where(bd["self_employed"]=="Yes",1,0)

del bd["self_employed"]


#Family income
bd["family_income"].value_counts()


#Checking the mean sensitivity towards y column
bd.groupby("family_income")["y"].mean().round(2) #This is the equivalent of tapply in R

#Doing the trasnformation
bd["fi"]=4
bd.loc[bd["family_income"]=="< 8,000, >= 4,000","fi"]=6
bd.loc[bd["family_income"]=="<10,000, >= 8,000","fi"]=9
bd.loc[bd["family_income"]=="<12,500, >=10,000","fi"]=11.25
bd.loc[bd["family_income"]=="<15,000, >=12,500","fi"]=13.75
bd.loc[bd["family_income"]=="<17,500, >=15,000","fi"]=16.25
bd.loc[bd["family_income"]=="<20,000, >=17,500","fi"]=18.75
bd.loc[bd["family_income"]=="<22,500, >=20,000","fi"]=21.25
bd.loc[bd["family_income"]=="<25,000, >=22,500","fi"]=23.75
bd.loc[bd["family_income"]=="<27,500, >=25,000","fi"]=26.25
bd.loc[bd["family_income"]=="<30,000, >=27,500","fi"]=28.75
bd.loc[bd["family_income"]==">=35,000"]=35

bd=bd.drop(["family_income"],1)



#Dropping any NAs in our data set
bd.dropna(axis=0,inplace=True)

bd.dtypes

#Splitting the data into train and test
bd_train, bd_test = train_test_split(bd,test_size=0.2,random_state=2)


#Dropping the unecessary variables
x_train=bd_train.drop(["y","REF_NO"],1)
y_train=bd_train["y"]
x_test=bd_test.drop(["y","REF_NO"],1)
y_test=bd_test["y"]


#Setting up or initiating the logistic regression function
logr=LogisticRegression(penalty="11",class_weight="balanced",random_state=2)
#class_weight="balanced" is an attribute that must be given for if our classes are imbalanced, then this param
#would help balance out our classes. (Note that ifthe classes of the dependent variable are balanced, then we 
#don need to give such an attribute, however it's better to give this attribute always just to be sure) 


#Fitting the data
logr.fit(x_train,y_train)


#score model performance on the test data (AUC curve)
roc_auc_score(y_test,logr.predict(x_test))



