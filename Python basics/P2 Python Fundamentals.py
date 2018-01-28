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
    print('value for key:%s is' % (elem),":",d[elem])

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

