# -*- coding: utf-8 -*-
"""
Created on Sun Jan 28 12:51:40 2018

@author: Abhinaba
"""

#Data handling with Python
#Dictionary
#Declaring a Dictionary (All values are stored as key value pairs)
#Data Dictionaries are optimised for search and querying
d={"actor":"nasir","animal":"dog","earth":1,"list":[1,2,3]}

#Checking the data type of the values in a Dictionary
type(d) #We would get dict that represents a Dictionary

#We cannot access the elements of a Dictionary with index, we rather need to refer to their keys
#e.g.
d["animal"] #We would get the value dog if we run this query

#We can also change values to the elements in a dictionary in the following way
d["animal"]="tiger" #Assigning the value tiger
d["animal"] #checking the value in the dictionary

#Deleting a key value pair in a dictionary
del d["animal"] #Deleting the key value pair having the key as animal
d #Seeing the dictionary to verify the same

#In case we pass a new value to a dictionary, we can create a new key value pair
d["species"]="Human" #Appending a new key value pair in the existing dictionary
d

#Iterating through a dictionary
for elem in d:
    print('value for key: %s is' % (elem),":",d[elem])

#A shortcut to see the items of the dictionary is follwoing
d.items() #It bascially gives us both key and value pairs and it returns a list of tuples
          #Each key value pair is going to be a tuple
          
#d.items can also be used to iterate through the elements of a dictionary in the following way
for a,b in d.items():
    print('value for key : %s is %s' % (a,b))


#We can also create place holders for any of our texts, say for e.g.
print('we are in %s session from %s and this is our %d class' % ('python','edvancer',3))
#or way 2
print('We '+ str(' Python ')+str(1)+' end')

#--------------------------------------------------------------------------------------------------

#Sets
#A set is basicallly a list or a collection of distinct objects
#e.g.
animals = {'cat','dog'} #Note that all the elements in this list have to be distinct

#We can check if a particular value is present in the list or not using the 'in' operator
#(It returns a boolean variable)
'cat' in animals #This would return true

'fish' in animals #This would return true
 
#Adding values to the set animals
animals.add('fish') 
print(animals)

#Adding values to the set animals
animals.add('cat') #If we try and run this we won't get any error, however, we would not be able 
#add it as it's already present in the set
#We can check it by runing the set below 
print(animals)
   
#Removing elements from Sets
animals.remove('cat')
print(animals)
#If we try to remove the 'cat' from the set again, we would get an error as the cat doesn't exist
#in the set  

animals={'cat','dog','fish','here','there'}
for animal in animals:
    print(animal)
#Note that running a loop through the elements of a set will not give the output by the order in
#which they are defined, the output can be random


#set operations like union, intersection, subset, superset etc
a={1,2,3,4,5,6}    
b={4,5,6,7,8,9}    

#union
c=a.union(b)
c

#intersection
c1=a.intersection(b)
c1

#checking subset (Checking if a is a subset of b) [Will return a boolean value]
a.issubset(b) #return false in this case

#checking subset (Checking if a is a subset of c) [Will return a boolean value]
a.issubset(c) #return true in this case

#checking superset (Checking if a is c is a superset of b) [Will return a boolean value]
c.issuperset(b) #return true in this case

#Difference operation (This would remove the elements that are already there in the 1st
#set)
a
b
a.difference(b) #We would get the elements that are there in a but not in b
b.difference(a) #We would get the elements that are there in b but not in a


#Symmetric difference (it will basically give an output excluding the common elements in a and b)
a
b
a.symmetric_difference(b) #You will see that the common elements 456 are not there in this list


#At times we need to cast a list to a set to avoid duplication in our records
#e.g.
li=[1,2,2,3,4,5,6,4,1,7,8] #This is a list containing duplicate values
print(li)
print(type(li))
lout=list(set(li))
print(type(set(li)))
print(lout)
print(type(lout))


#--------------------------------------------------------------------------------------------------

#Tuples
#Tuples are immutable lists, immutable means that we cannot change it's elements
#Just like data dictionaries, tuples are also optimised for search and query kind of experiences
t=12345,54321,'hello'
t

#The above can also be writtn in parenthesis
t=(12345,54321,'hello')
t
#If we declaring in any of the following way, then it will automatically considered as a tuple data struc.

#Also, a tuple can contain any type of object, even a tuple itself. e.g.
u=t,'abc',123
u


#You cannot reassign variables in a tuple, that is, you cannot do something like
t[0]='abc'  #This is coz tuples are immutable data types

#The advantage of tuple over list is that since it's immutable and has a fixed size, it can be most widely
#used in pure operations or functions where the original object has to circulated without changing any of
#its values (One of the applications of this is network programming). Also, it's much more faster than list
#when the data volumes are very high.


#Also, one more important feature of tuples is that even though a tuple is immutable, but the elements of the
#tuple may or may not be immutable, that is we can change their values(this depends on the underlying data structure of the object).
#e.g.
v=([1,2,3],[3,2,1])
v[0][0]=21
v



#Also, one of the most important diff.in tuples and lists is that tuples can be used as keys in dictionaries and
#as elements of sets, while lists cannot

#Creating a dictionary with tuple keys
d={(x,x+1): x for x in range(10)} #We are basically doing dictionary comprehension
d
#Seeing the elements of the dictionary (Calling the 6th element)
d[(5,6)]
#We can also try the following
t=(5,6)
print(type(t))
print(d[t])
#So, key for a dictionary can be an integer, string but cannot be a list or floating point number

#--------------------------------------------------------------------------------------------------
#We are impoting the library math
import math


#Object Oriented Programmming with Python
class Point(object): #Our class is inherited from the super class object
    
    def __init__(self, x,y): #This is a constructor for the class Point
        '''Defines x and y variables''' 
        self.X=x   #Object variable X
        self.Y=y   #Object variable Y
        #Self is used as reference to internal objects that get created using values supplied from outside.
        #Esentially, self pointer is a reference to the location in the memory where the object is located.
        #default function __init__ is used to create internal objects which can be accessed by other functions
        #in the class
        
    def length(self):  #Object method, this is used to perform a specific set of actions
        return(self.X**2+self.Y**2)
        
        #All functions if a class will have access to it's internal objects created inside the class, you can
        #interpret self to be default Point class object
    
    def distance(self, other):
        dx=self.X - other.X
        dy=self.Y - other.Y
        return math.sqrt(dx**2 + dy**2)
        #A function inside a class can take input multiple objects
        #of the same class
        

#Calling the above class objects
z=Point(2,3) #We do not need to pass a value for Self, as it would be called implicitly
z.X #This would return 2
z.Y #This would return 3

#Calling the method length, note that self would be implicitly called
z.length() #This would return 13, that is x square + y square

#Calling the method distance with instantiation of another object y, e.g.
y=Point(5,6)
#Calling the function distance with the instantiation of another object y, self would be implicitly called
z.distance(y) #This would give us 4.2426
#i.e; sqrt(18)

#We can also apply function on and between objects of this class Point by using syntax clas_name.function.
#example given below
Point.length(z)
Point.length(y)

#--------------------------------------------------------------------------------------------------
#Data Handling in Python

#At times, what happens is that a lot of libraries that are used in Python are not updated, and they may
#throw warnings, so in order to avoid that, what we do is that we import the library warnings and filter out
#the warnings in the following way
import warnings
warnings.filterwarnings('ignore') #This would help us get warnings again and again



#Importing numpy library, it's used as a numerical Python library
import numpy as np
import math


#Matrix
#Creaet a 2D array or Matrix
b=np.array([[1,2,3],[4,5,6]])
b

#We can determine the dimension of an array with the following attribute
b.shape #this would return (2,3), shape is an attribute of the class numpy array

#Fetching the elements of a matrix
b[0,0] #refers to first row, first column... this would give 1
b[1,2] #refers to second row, 3rd column... this would give 6
b[1,0] #refers to second row, 1st column... this would give 4
b[2,0] #refers to third row, 1st column... this would give an error coz our matrix is a 2x3 Matrix


#Qucik way to initialise a matrix or numpy array with all 0s or all 1s
#e.g.
#All 0s
np.zeros((2,4))
#All 1s (This is also called an intercept)
#intercept is defined as all 1s column in a data frame or numpy array 
np.ones((2,4))

#We can also declare matrix or numpy array with varied or irregular lengths, e.g.
a=np.array([[1,2,3],[4,5]])
#In this case a matrix would be created with irregular size, we can verify the same
#using the shape function
a.shape #Here we would get (2,), since the column size is unknown, this is helpful while applying
        #neural networks or deep learning as there we need dynamic arrays
#Most of the places where we have structured data, we use normal nupy arrays, however, whenever we
#have to use unstructured data like speech, video or text, we need dynamic or unstructured numpy arrays
        
        
#3-D matrix
a=np.array([[[1,1],[2,2],[3,3]],[[4,4],[5,5],[6,6]]])        
a
#If we do a shape of the above syntax, we get (2,3,2)
a.shape        

#Creating all constants with numpy array
np.full((2,2),math.pi)


#Creating all constants along with defining data type
np.full((2,2),4,dtype=int)


#Creating an identity matrix with numpy
np.eye(5) #Application: Single Value matrix


#Using random numbers from numpy to fill up numpy arrays
np.random.random((4,3))

#Also, single random numbers from [0,1] can be obtained like this
np.random.random()


#e.g. 2, random numbers from [5,95]
90*np.random.normal()+5

#or
np.random.normal(0,1)

#So, we see various ways of using np.random, there are also some other ways of using np.random and the
#above examples, do not cover all the scenarios


#You can also use a seed for your random selection, say for e.g.
np.random.seed(7)
