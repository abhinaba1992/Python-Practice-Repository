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
#Now, if we do a[c], then we would basically get the same result as simply running a because since c 
#contain [0,1,2,3], running a[c] would mean runing a[0,1,2,3], that would return all the four rows and
#all cols, so we would bascially get the same result
a[c]


#Als, if we do the following. We would still get the same result, as we would get all the first four rows
a[:4]

a[c,b] #Here all the rows are selected, however specific cols are selected from each row

a[:4,b] #Similiar ways of achieving the above

