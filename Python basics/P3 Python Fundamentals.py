# -*- coding: utf-8 -*-
"""
Created on Mon Mar 12 09:55:34 2018

@author: Abhinaba
"""
import numpy as np
import math


#Defining a numpy array
a= np.array([[1,2,3,4],[5,6,7,8],[9,10,11,12]])
#So, basically we define a 3X4 array in the above line

#Subsetting the array
b=a[:2,1:3] #So, the 1st and 2nd rows & 2nd and 3rd cols will be subsetted

b #note that b is still a part of array a, i.e.; it references a. any changes in b, will still affect a

#However, if you wanna create a new reference to the memory, you need to do the following
c=a[:2,1:3].copy() #Using the keyword copy for creating a new reference in the memory

c  #This is a new reference and making changes to it, will not effect the original array a

#So, when subsetting an array, it's always a good practice to use the keyword copy

#Way to of subsetting the array
a= np.array([[1,2],[3,4],[5,6]])
#3X2 array

a[[0,1,2],[0,1,0]] #Here we are directly specifying the number of rows and cols for subsetting i.e.; we are 
#specifying all the 3 rows and the 1st col, 2nd col and 1st col respectively for the data sets. i.e; we can 
#choose different cols in a non-monotonic way here (this is not possible in R, this is specific to python)

#Another example
a[-3:-1] #So basically, the first row and second row is picked up here since the first row is also indexed 
#at -3 and second row is indexed at -2, so our subsetting will work for the first and second row only

a[:3] #All the first 3 rows i.e.; 0, 1 and 2

a[:3,:2] #All the first 3 rows i.e.; 0, 1 and 2. and first two cols i.e.; 0 and 1


#Example with another array
a=np.array([[1,2,3],[4,5,6],[7,8,9],[10,11,12]])
#4x3 array.

a.shape

#Another array
b=np.array([0,2,0,1])
c=np.arange(4) #This is another way of defining an array using numpy, this is almost like a for loop,
               #the only difference is that it doesn't return a list/range, but a numpy array itself
c

#Also, in case we want to define a range for arnage, we can also do that, so for e.g., if we do
c=np.arange(0,4,1) #Start at 0, go for 4 elements, increment by 1
c
#Now, if we do a[c], then we would basically get the same result as simply running a because since c 
#contain [0,1,2,3], running a[c] would mean runing a[0,1,2,3], that would return all the four rows and
#all cols, so we would bascially get the same result
a[c]


#Als, if we do the following. We would still get the same result, as we would get all the first four rows
a[:4]

a[c,b] #Here all the rows are selected, however specific cols are selected from each row

a[:4,b] #Similiar ways of achieving the above


#Other example definitions with np.arange
#e.g. 1 (We create an array in this step)
n=np.array(np.arange(5))
n
#Checkout the shape
n.shape

#e.g. 2 (We create a matrix in this step)
n=np.matrix(np.arange(5))
n
#Checkout the shape
n.shape 
#Numpy matrix can only have 2 dimensions, while numpy array can have as many dimensions as possible

#Changing values in a numpy array
a[c,b] += 10 #Adding 10 to the selected elements (We can also perform actions like multiplication, division 
a             #etc)



#Conditional subsetting in numpy arrays
a=np.array([[1,2],[3,4],[5,6]])
#Subsetting with a condition
c=a>2 #c will have a array of the same shape as a and would contain TrueFalse values
#seeing c value
c

#Subsetting a only when cases are True (based on the above example)
a[c] #We would only get those values where c is true
#or
a[a>2]


a[c].shape #this will givve us 4 since we got 4 TRUEs


#More complex subsetting ops.
a[(a>2) | (a<5)] #or operation
a[(a>2) & (a<5)] #and operation


#Array operations
#it's important to note that the type of input for a method/operation in a particular library can be either 
#a numpy array or a numpy matrix depending upon how it's developed (so whatever representation is expected 
#by our library, we have to put or data in that format)

#Defining 2 arrays, X and Y
X=np.array([[1,2],[3,4]])
Y=np.array([[5,6],[7,8]])

#Note that whatever array is larger, the final shape would be according to that
#e.g.
L=np.array([[1,2],[3,4]])
M=np.array([[5,6]])

L+M, L-M, L*M, L/M
M+L, M-L, M*L, M/L

#Addition
X+Y
#We can also use numpy add function to do the same
np.add(X,Y)

#Substract
X-Y
#using numpy function
np.subtract(X,Y)

#Multiplication
X*Y
#Using numpy function
np.multiply(X,Y)

#Division
X/Y
#Using numpy function
np.divide(X,Y)

#Square root
np.sqrt(X)
#note that numpy is derived from math, so if we do not import numpy, then we cannot perform these operations
#on numpy array. math library allows performing sqrt on int or float values, however in order to do the same
#on data structures like numpy arrays, we need the library numpy (implicitly math is a supeclass for numpy)


#matrix multiplication using dot function
V=np.array([9,10])
W=np.array([11,12])

#Using the dot operator below
V.dot(W) #We would get 219 ((9*11) + (10*12)

#OR
V=np.array([[7,8],[9,10]])
W=np.array([[11,12],[13,14]])

V.dot(W) #We would get 219 ((9*11) + (10*12)


#We cannot use dot operator for matrix mul. in case the number of elements in a row is not equal to the equi.
#column of the matrix
V=np.array([9,10])
W=np.array([11])

V.dot(W) #We would get an error while trying to this.
#Or
V=np.array([[7,8],[9,10]])
W=np.array([[11,12],[13,14],[15,16]])

V.dot(W) #We would get an error


#Transposing a matrix
V=np.array([[7,8],[9,10]])
W=np.array([[11,12],[13,14],[15,16]])
#Transposing
V.T
W.T

#So, the use of the transpose here is that in order to make sure that the dot product works, sometimes we may
#need to make sure that the matrix is transposed in such a way that the no. of elements in a column matches
#that of the rows in another matrix


#Summing up the elements in a numpy array
np.sum(X) #WE would get 10

#Summing across dimensions
np.sum(X,axis=0) #Across cols
np.sum(X,axis=1) #Across rows


#Creating copies of a elements in an array using tile function (It is basically used for copying a similar
#structure across many number of rows or cols)
v=([1,0,1]) #Defining an array V
np.tile(v,(4,1)) #Creating 4 copies having the same elements
np.tile(v,(4,2)) #Creating 4 copies having twice the number of elements


#--------------------------------------------------------------------------------------------------
#Panda Data Frames
import pandas as pd
#This is one of the most important libraries in Python, this is used for data manipulation & Handling
#and manipulating data

Cities=["Delhi","Mumbai","Kolkata","Chennai"]
Code=[11,22,33,44]


mydata=list(zip(Cities,Code))
#We would get a cross product of cities and pin codes

#Creating a data frame in Panda
df=pd.DataFrame(data=mydata,columns=["cities","codes"])
df #We would get a data frame with two cols
#Also, if we din't give any column name, it just assumes 0 and 1 to be it's column name


#Writing a data frame to a file (CSV file)
df.to_csv("mydata.csv",index=True,header=True)
#Note that we should be careful while writing the file with index (or row numbers) because if we are trying to
#read back the same file again, then the index column would be made into a new column which is unecessary.

#Reading a data frame from a CSV file
df_temp=pd.read_csv("mydata.csv")

df_temp #Here we would find that the index column has formed a new column and also a new index/rowname column
        #is developed, so it's advisable not to use indexing while writing the data into the path
        
#Excel writer
writer=pd.ExcelWriter("mydata.xlsx")       
df.to_excel(writer,"Sheet1",index=True)
df.to_excel(writer,"Sheet2")


#Reading a file from a location predefined location
myfile=r'C:\Users\Abhinaba\Desktop\Edvancer Materials\Python\Data\bank-full.csv'
#Choosing the selected cols we want and specifying the seperator
bd=pd.read_csv(myfile,usecols=["age","job","marital","balance"],sep=";")

#Checking the data type of bd
type(bd) #This will return pandas.core.frame.DataFrame

#Checking the type of col or series
type(bd['job']) #This will return pandas.core.frame.Series

#Seeing all the columns of a data set
bd.columns

#Seeing the top 10 rows
bd.head(10)

#Seeing the last 10 rows
bd.tail(10)

#Creating cross tabulation (Margins here is used to get the sum)
pd.crosstab(bd["job"],bd["marital"],margins=True)
#If you dont want summation, just make margins equal to false
pd.crosstab(bd["job"],bd["marital"],margins=False)


#Aggregating by a feature value in cross tabulation
pd.crosstab(bd["job"],bd["marital"],values=bd["balance"],aggfunc=np.mean)

#Doing group by operation in Python
bd.groupby("job")["balance"].mean()  #equi. to: select mean(balance) as bmean from <table name> group by job


#Each row in a data frame is known as a series, thus multiple series in a data frame combine together to give
#a data frame.

#Value counts for multi level categorical data
bd.groupby("job")["marital"].value_counts()  #equi. to: select job, count(marital) as mcnt from <table name> group by job,marital

#Value counts for single tier categorical data
bd["marital"].value_counts() #Equi. to select marital, count(martical) as mcnt from <table name> group by marital

#Mean for single tier numerical data
bd["age"].mean()
#or
bd["balance"].mean()


#Seeing the data types of all the cols in data frame
bd.dtypes

#Note that the data type of a particular feature in Python is decided by looking upon the first few thousand rows
#of a feature, anything which is integer is denoted as int64, while others are denoted as object. Note that in case
#We have an int feature that contain a NAN value in first few thousand rows, in that case, we will get a object data
#type even for an int column, which won't be a case if the NAN value lies much later in a data frame. So, its always
#a good idea to cross check the data types in a data frame because it looks only at the top few thousand records

#We can also check the data type of a particular feature in the following way
bd["marital"].dtype
#We will get dtype('o') that denotes that it's an object type


#We can also verify if a feature belongs to a particular data type in the following way
bd["marital"].dtype=='O' #This will return True (would return false if marital was of some other data type)


#gives us a description of the numeric columns
bd.describe()
#Find describe by specific columns
#1 cols
bd["age"].describe()
#or way 2
bd.age.describe() #This syntax is good only for enquiring the result and not for assignment

#multiple cols
bd[["age","balance"]].describe()
#In the same way we can use other functions like mean, median, mode, max, min, sd etc


#Subsetting
#Subsetting fields from a data set based on column names
bd[["age","balance"]]

#Subsetting based on data types (We will get all the fields that are categorical in nature)
bd_cat_data=bd.select_dtypes(['object'])
#Seeing the same
bd_cat_data.columns #We will get the cols with object data type in this data set


#Using a for loop for doing a value count for each of my categorical variable, we do the following
for c in bd_cat_data.columns:
    print(bd[c].value_counts())
#We will get counts of all the categorical variables for all the cols
    
#Using for loop to get the number of unique categorical variables for each cols
for c in bd_cat_data.columns:
    print(c,":",bd[c].nunique())
#nunique function gives us the number of unique categorical variables for a feature, it can also be used
#directly for a column    


#to get the count of rows and cols in Python, we use shape
bd.shape


#To get the median of all the numerical cols, we use the following
bd.median()


#Reimporting the bank data with all cols
bd=pd.read_csv(myfile,sep=";")
bd


#Multi tier group by operations (similar to the ones shown earlier above)
bd.groupby(['housing','loan'])["age","balance"].mean()

#with value countss for categorical variables
bd.groupby(['housing','loan'])["education"].value_counts()


#Doing different operations using group by clause for different fields
bd.groupby(['housing','loan']).agg({'age':'mean','duration':'max','balance':'sum'})
#Whatever agg functions we use, must come under the numpy package, like for eg mean, max and sum should fall
#under numpy, also aggregate function consumes a dictionary where key is the col name and value is the function
#name


#Plotting values of a field
bd["balance"].plot()


#Using ggplot for data Vizulaistion
#    conda install -c https://conda.binstar.org/bokeh ggplot
from ggplot import *

