# -*- coding: utf-8 -*-
"""
Created on Mon May 14 10:15:08 2018

@author: Abhinaba
"""

#We are using the same example we sued for the logistic regression part, hence we are copying that entire part
#of data importing and data preparation from logistic regression file (Only the importing of libraries is diff.)

#Decision Tree

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


#Now the objective is to find the best tree size using the F2 score as the criterion for cross validation
max_nodes=[5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22]
max_nodes


#We are doing cross validation here
beta=2 #The more the beta value goes towards infinity, the F2 score becomes recall, while the more it goes
       #towards 0, the F2 score becomes precision [(0-1):Precision,(2-infinity):Recall)
FB_avg=[]
for max_node  in max_nodes:
    mytree = tree.DecisionTreeClassifier(criterion="entropy",
                                         max_leaf_nodes=max_node,class_weight="balanced")

    # computing average RMSE across 10-fold cross validation
    kf = KFold(len(x_train), n_folds=10)
    FB_total = []
    for train, test in kf:
        mytree.fit(x_train.loc[train], y_train[train])
        p = mytree.predict(x_train.loc[test])
        df=pd.DataFrame(list(zip(y_train,p)),columns=["real","predicted"])
        TP=len(df[(df["real"]==1) &(df["predicted"]==1) ])
        FP=len(df[(df["real"]==0) &(df["predicted"]==1) ])
        TN=len(df[(df["real"]==0) &(df["predicted"]==0) ])
        FN=len(df[(df["real"]==1) &(df["predicted"]==0) ])
        P=TP+FN
        N=TN+FP
        Precision=TP/(TP+FP)
        Recall=TP/P
        FB=(1+beta**2)*Precision*Recall/((beta**2)*Precision+Recall)
        FB_total.extend([FB])
    FB_avg.extend([np.mean(FB_total)])
best_max_node=np.array(max_nodes)[FB_avg==max(FB_avg)][0]

print('max_node value with best F2 score is :',best_max_node)
#We get 15



#We are now re-running the dtree with the optimised number of max nodes we ot above (which is 15) 
dtree=tree.DecisionTreeClassifier(criterion="entropy",
                                  max_leaf_nodes=best_max_node,class_weight="balanced")
dtree.fit(x_train,y_train)
predicted=dtree.predict(x_test)

df_test=pd.DataFrame(list(zip(y_test,predicted)),columns=["real","predicted"])

k=pd.crosstab(df_test['real'],df_test["predicted"])
print(k)


#Checking the metrics
TP=k.iloc[1,1]
TN=k.iloc[0,0]
FP=k.iloc[0,1]
FN=k.iloc[1,0]
P=TP+FN
N=TN+FP

print(TP,TN,FP,FN)
print('Accuracy is :',(TP+TN)/(P+N))
print('Sensitivity is :',TP/P)
print('Specificity is :',TN/N)


#===================================================================================================
#===================================================================================================

#Random Forest


import numpy as np

from time import time
from operator import itemgetter
from scipy.stats import randint as sp_randint
from sklearn.ensemble import RandomForestClassifier
from sklearn.grid_search import RandomizedSearchCV


clf = RandomForestClassifier(verbose=1,n_jobs=-1)
#verbose=1 would let us see the intermediate outputs we can get
#n_jobs can be passed based on the number of cores  in our CPU based on which the analysis would be
#done, it basically considers the parallel processing capability that can be achieved, if we do not know
#the number of cores, it's safe to set n_jobs=-1 so that it can automatically figure out the number of cores.


#Below we have a Utility function to report the best scores. This simply accepts grid scores from our
#randomSearchCV/GridSearchCV and picks and gives top few combination according to their scores
#In other words the following is a helper function that gives us the rank of a model based on mean validation
#scores for a set of parameters


def report(grid_scores, n_top=3):
    top_scores = sorted(grid_scores, key=itemgetter(1), reverse=True)[:n_top]
    # above line selects top n grid scores
    # for loop below , prints the rank, score and parameter combination
    for i, score in enumerate(top_scores):
        print("Model with rank: {0}".format(i + 1))
        print("Mean validation score: {0:.3f} (std: {1:.3f})".format(
              score.mean_validation_score,
              np.std(score.cv_validation_scores)))
        print("Parameters: {0}".format(score.parameters))
        print("")


#In the following randomSearchCV/GridSearchCV accept parameters values as dictionaries
#In example given below we have constructed dictionary for different parameter values
#that we want to try for randomForest model (We are setting up the grid search parameters here)        

param_dist = {"n_estimators":[10,100,500,700],
              "max_depth": [3,5, None],
              "max_features": sp_randint(5, 11),
              "min_samples_split": sp_randint(5, 11),
              "min_samples_leaf": sp_randint(5, 11),
              "bootstrap": [True, False],
              "criterion": ["gini", "entropy"]}
#Total number of combinations we would get from above is 4 X 3 X 7 X 7 X 7 X 2 X 2 which would take a lot
#of time to run and hence we would need to randomly choose certain combinations from the above


#Running the randomized Search (We are randomly picking 100 combinations out of the above mentioned 
#combinations)
n_iter_search = 100


#n_iter parameter of RandomizedSearchCV controls, how many parameter combination will be tried; out
#of all possible given values, we are also setting the  param_distributions parameter as param_dist
#Which contains all the combinations of parameters
random_search = RandomizedSearchCV(clf, param_distributions=param_dist,
                                   n_iter=n_iter_search)
random_search.fit(x_train, y_train)
report(random_search.grid_scores_)



#Based on the best set of features we got, we are building our random forest classifier with the 
#set of values we got from the RandomizedSearchCV function
rf=RandomForestClassifier(n_estimators=700,verbose=1,criterion='entropy',min_samples_split=7,
                         bootstrap=False,max_depth=None,max_features=8,min_samples_leaf=8,
                          class_weight="balanced")


#Fitting the entirety of our train data on the following set of values
rf.fit(x_train,y_train)

#doing the prediction on the test
predicted=rf.predict(x_test)

df_test=pd.DataFrame(list(zip(y_test,predicted)),columns=["real","predicted"])

#Finding the cross tabulation between the real and the predicted
k=pd.crosstab(df_test['real'],df_test["predicted"])
print(k)


#Calculating the matrix
TP=k.iloc[1,1]
TN=k.iloc[0,0]
FP=k.iloc[0,1]
FN=k.iloc[1,0]
P=TP+FN
N=TN+FP

#Printing all the matrix
print(TP,TN,FP,FN)
print('Accuracy is :',(TP+TN)/(P+N))
print('Sensitivity is :',TP/P)
print('Specificity is :',TN/N)


#Following we see feature importances, the way feature importance is calculated is the importance of a 
#feature is dependent on the number of times it has been utilised for a split, so here we would get values
#between 0.0 to 1.0, i.e. percentages of values 


importances = rf.feature_importances_
std = np.std([tree.feature_importances_ for tree in rf.estimators_],
             axis=0)
indices = np.argsort(importances)[::-1]

#Print the feature ranking
print("Feature ranking:")

for f in range(x_train.shape[1]):
    print("%d. feature %s (%f)" % (f + 1, list(x_train.columns)[f], importances[indices[f]]))

#Plot the feature importance of the forest
plt.figure()
plt.title("Feature importances")
plt.bar(range(x_train.shape[1]), importances[indices],
       color="r", yerr=std[indices], align="center")
plt.xticks(range(x_train.shape[1]), list(x_train.columns))
plt.xlim([-1, x_train.shape[1]])
plt.show()



#Partial Dependence plot

#One big difference we find in these algorithms is that we don't have straight forward coefficients associated
#with variables. We dont have a clear picture of how a particular variable affects our response, other than 
#saying that some variable is important and some are not

#We can solve this issue by building partial dependency plot ourselves. Sklearn currently has partial dependence
#plot support only for gradient boosting machines. However, we can implement a basic version of dependence plot
#ourselves.


#The idea is to fix all other variables and vary particular predictor in question and see how the predicted response
#values move. [A better implementation will be where we randomly sample other variables instead of fixing them to
#a fixed value but that will take a lot of time and code, so we need to keep it simple for now]


#We'll do this for the variable children as it has one of the highest importance (We can also check the 
#same for other variables for our reference)
#We can try to generalise the process and even write a function if we want

data=x_train.copy()

features=x_train.columns

for f in features:
    if f=='children':pass
    else:
        data[f]=data[f].mean()

data=data.drop_duplicates()
data['response']=pd.Series(list(zip(*rf.predict_proba(x_train)))[1])



from ggplot import *

ggplot(data,aes(x='children',y='response'))+\
geom_line(colour='steelblue')+\
xlab("Children")+\
ylab('Response')+\
ggtitle('Partial Dependence Plot \n Response Vs Number of children')
