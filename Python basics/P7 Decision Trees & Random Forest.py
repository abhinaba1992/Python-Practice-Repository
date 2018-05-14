# -*- coding: utf-8 -*-
"""
Created on Mon May 14 10:15:08 2018

@author: Abhinaba
"""

#We are using the same example we sued for the logistic regression part, hence we are copying that entire part
#of data importing and data preparation from logistic regression file (Only the importing of libraries is diff.)

import pandas as pd
import numpy as np
import math as ma
from sklearn.cross_validation import train_test_split
from sklearn import tree
from sklearn.cross_validation import KFold
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')#This would help us ignore all warnings



#Data Preparation part is as is from logistic regression

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


#Self employed partner
bd["self_employed_partner"].value_counts()

#Doing the trasnformation
bd["semp_part_yes"]=np.where(bd["self_employed_partner"]=="Yes",1,0)

del bd["self_employed_partner"]


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

#Splitting the data into train and test
bd_train, bd_test = train_test_split(bd,test_size=0.2,random_state=2)


#Dropping the unecessary variables
x_train=bd_train.drop(["y","REF_NO"],1)
y_train=bd_train["y"]
x_test=bd_test.drop(["y","REF_NO"],1)
y_test=bd_test["y"]

#Resetting the index in the x_train and y_train data Set
x_train.reset_index(drop=True,inplace=True)
y_train.reset_index(drop=True,inplace=True)


#Creating a decision tree class object (We are using DecisionTreeClassifier here because we want to solve 
#a classification problem here)
dtree=tree.DecisionTreeClassifier(criterion="entropy",max_leaf_nodes=10,
                                  class_weight="balanced")
#class_weight="balanced" is an attribute that must be given for if our classes are imbalanced, then this param
#would help balance out our classes. (Note that ifthe classes of the dependent variable are balanced, then we 
#don need to give such an attribute, however it's better to give this attribute always just to be sure) 
#Criterion can either be Entropy or Geni Index, in Python we usually go for Entropy, and max leaf nodes=10
#tells us that there would 10 number of terminal nodes


#Fitting the data on the tree
dtree.fit(x_train,y_train)


#Exporting a graph Viz object, please note that we need to do this in order see the vizualisation of the 
#dtree (graphviz is an independent software that needs to be donwload seperately)
tree.export_graphviz(dtree,out_file=r"C:\Users\Abhinaba\Desktop\Edvancer Materials\Python\Data\mytree.dot",
                     feature_names=x_train.columns,
                    class_names=["0","1"],
                     proportion=True)
#The exported dot file here is just the schema that describes how the tree looks like 
#(The file would give us the decision tree rules in simple text format, .dot file can be opened by a simple
#text editor.)


#Making predictions using the tree
predicted=dtree.predict(x_test)

df_test=pd.DataFrame(list(zip(y_test,predicted)),columns=["real","predicted"])

k=pd.crosstab(df_test['real'],df_test["predicted"])


#Seeing the confusion matrix
k


#Getting the confusion matrix
TP=k.iloc[1,1]
TN=k.iloc[0,0]
FP=k.iloc[0,1]
FN=k.iloc[1,0]
P=TP+FN
N=TN+FP

#Checking the matrix
print('Accuracy is :',(TP+TN)/(P+N))
print('Sensitivity is :',TP/P)
print('Specificity is :',TN/N)
print('Precision is :',TP/(TP+FP))




