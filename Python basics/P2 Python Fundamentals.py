# -*- coding: utf-8 -*-
"""
Created on Sun Jan 28 12:51:40 2018

@author: Abhinaba
"""

#Data handling with Python
#Dictionary
#Declaring a Dictionary (All values are stored as key value pairs)
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


