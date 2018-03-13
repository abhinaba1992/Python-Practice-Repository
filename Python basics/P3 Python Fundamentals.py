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
