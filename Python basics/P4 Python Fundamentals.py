# -*- coding: utf-8 -*-
"""
Created on Wed Mar 28 11:05:14 2018

@author: Abhinaba
"""

import pandas as pd
from ggplot import *
import math as ma

myfile=r'C:\Users\Abhinaba\Desktop\Edvancer Materials\Python\Data\bank-full.csv'

bd=pd.read_csv(myfile,sep=";")

bd

#Drawing a box plot we drew in P3
ggplot(aes(y="age",x="education"),data=bd)+geom_boxplot()


#Drawing a histogram with ggplot
ggplot(aes(x="balance"),data=bd)+geom_histogram()+ggtitle("Histogrm")

#Histogram with custom bin size
ggplot(aes(x="balance"),data=bd)+geom_histogram(bins=100)+ggtitle("Histogrm")

#Density Plot
ggplot(aes(x="balance"),data=bd)+geom_density()+ggtitle("Density Curve")

#Bar plot
ggplot(aes(x="marital"),data=bd)+geom_bar()+ggtitle("Bar plot for categorical variables")

#Bar plot with multiple categories
ggplot(aes(x="marital",fill="housing"),data=bd)+geom_bar()+ggtitle("Bar plot for categorical variables")

#Showing data manipulation in Python
cities=["Delhi","Mumbai","Kolkata","Chennai"]
code=["11","22","33","4a"]

mydata=list(zip(cities,code))
df=pd.DataFrame(data=mydata,columns=["cities","codes"])

df

#Checking the types
df.dtypes

#In case our series contains all the numeric numbers we can convert it into numeric in the following way
df["codes"]=pd.to_numeric(df["codes"]) #This will give an error coz we have 4a as one of the value for codes


#In case we want to convert the column to numeric anyways with the non nnumeric values specified as NA, we can
#do the following 
df["codes"]=pd.to_numeric(df["codes"],errors="coerce")
df #We will get NaN for non numeric values, also because we have NaN in one of the values the data type of the 
   #Series will become float, as NaN belongs to float data type, so it will take presedence over int
   


#Feature Engineering with Python
df["cities2"]=[X.replace("a","6") for X in df["cities"]]   
df


df["code_log"]=[ma.log(X) for X in df["codes"]]   
df