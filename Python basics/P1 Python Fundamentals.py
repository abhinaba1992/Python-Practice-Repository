# -*- coding: utf-8 -*-
"""
Created on Sun Jan  7 07:47:57 2018

@author: Abhinaba
"""
#Pyhton part 1 training 24th Dec 2017
#Author: Abhinaba Chakraborty
#Date: 19th Jan 2018

#hash helps us to give single line comment
#We can provide comments also by selecting the entire line and pressing ctrl+/
#While we can give multi line comments using ''' and '''
'''
This is an example line statement 
enclosed in a multiline comment sample
'''
#Shortcut for executing a line is ctrl+enter (just like we have in R)
#Python is case sensitive
#Pythong allows extensive error and exception handling mechanisms
#Python files are stored with the extension py
#var names cannot start with a number or cannot have spaces
#We use libraries in Python that contains piece of predefined codes or functions that can be used in our code (much like packages in R)


#SIMPLE VARIABLE ALLOCATION
x=5
y="data science"

#IMPORTING LIBRARIES
#We can import a library by using the keyword "import" (NOTE: Keyword is an actual library here)
#e.g. 1 [This has to be ran from console located down right]
import Keyword #for importing a library named Keyword
#Now if we wish to import specific components from the library, we can do
Keyword.Kwlist #[This has to be ran from console located down right]

#e.g. 2
#We can also import parts of a library in the following way
from Keyword import Kwlist #[This has to be ran from console located down right]
#Now the above approach is useful when we do not have another component with the same name in another library, like for e.g. we cannot have
#"from Keyword import Kwlist" and "from <some other library> import Kwlist" in the same piece of code as it will cause problem, so its always
#a good practice to import the library first and then refer to specific portions of the same like shown in example 1

#e.g. 3
from Keyword import * #[This has to be ran from console located down right]
#This is very much similar to e.g. 1, however, the only difference is that in example 1 you have to refer to specific components by preceding
#them with the name of the library and a dot(e.g. Keyword.Kwlist), but in e.g. 3 you can directly include the components with the component 
#name(Kwlist)

#e.g. 4
import Keyword as kw #[This has to be ran from console located down right]
kw.Kwlist  #[This has to be ran from console located down right]

#VARIABLE TYPES
#string, integer and float
x=5 #integer
y="data science" #string
z=0.5 #float
#just like R, python is dynamically typed programming language, that means the data type of a variables is decided dynamically based on it's 
#type

#PRINTING THE VALUE OF A VARIABLE
print(x)

#CHECKING THE DATA TYPES
type(x) #This would return the data type of an object variable (for e.g. int in this case)
print(type(x)) #This would return something like <class 'int'> (it bascially shows the class of the variable type to which it belongs)

#HANDLING QUOTES WITH PYTHONG STRING DATA TYPES (34 min)



#TYPE CASTING IN PYTHON
x="5.0"
float(x) #Converted x into float
#or
int(float(x)) #Converting it into integer, but for that to work, we must first convert them into a float and then into an integer
#note that when we do int(5.1) or int(5.9), then the int function would not automatically round it off to the nearest decimal, it would rather
#just extract the integer part out of the of the variable and just show the outpur. so for e.g. int(5.1) will give us 5 and also int(5.9)
#would give us 5

#Also when we did float(x) above, we didn't actually type cast x as we need to assign the casted variable back to the original variables so
#as to commit the casting, so for the above example we would need to do the following
x="5.0"
x=float(x) # casting and assigning it back to the main variable x


#GETTING SIZES OF VARIABLES
import sys #[This has to be ran from console located down right]
x=1
sys.getsizeof(x) #Gives us the size occupied by the variable in the memory in bits


#NUMERIC OPERATIONS IN PYTHON
x=1.22
y=20

x+y
x-y
x*y
x/y

#We can also do some absolute operations like
2/3 #which would give us 0.666666666
int(2/3) #Which would would gice us 0, as it will only take up the integer part

#We can also do multiple operations in asingle line e.g.
x+2,y+3,x+y,x**(x+y)
#the above code would return something like (3.21999999988,23,21.22,68.00731157)
#this is known as a tuple
#If we do the following, it would say tuple
type((x+2,y+3,x+y,x**(x+y))) #We would get a tuple by running this
#Now the above is particularly useful when we wish to return multiple values from a single return statement

#We can also do something like
a,b,c,d=x+2,y+3,x+y,x**(x+y)
#In the above case a would be assigned with x+2, b with y+3, c with x+y, d with x**(x+y)

#Multiple assignments can also be done like
a,b,c,d=2,3,4,5
#however, the following would give an error because integer is not an iterable operator. In other words,
#the assignment operation expects multiple values for assigning, and it will give an error if it's just
#a single value
a,b,c=2

#On the contrary, we can do something like the following
a,b,c=[2,2,2] #However in this case, the values that are being assigned won't be considered as a tuple
              #but rather as a list of integer values
              
#So in a practical scenario we can write a function like the following
a,b,c,d= f(x,y) return x+2,y+3,x+y,x**(x+y) #This is just an example, original syntax has not been followed
#So it would create a function that takes in x and y as params and retunrs the values in chronological order


#In Python, x^Y doesn't work for exponential values, we need to write x**y,that is
x^y #is wrong and would give an error
x**y #is right


#In general for bracket related arithmatic evaluation, we use BODMAS rule(Bracket of Diviion, multiplication,
#addition and substraction)
#However in python we use a similar concept known as PEDMAS (parenthesis,Exponent,Multiplication,
#Division(module peration),Addition,Substraction)

z=(x+y)**x-y/x # eqn. within the bracket will be solved first, and then the exponentail part, followed by 
z              # multiplication part, then the division part and finally the substraction part.So, we will
               # get a value of z as 25.1621..
               

#Modular operations
x=5
y=2               
x%y               


#Logical operations
x=True
y=False

#The follwoing two would give us datatype of the above variables
type(x)
type(y)

#Boolean operations (the simpel AND and OR functions)
x and y #FALSE
x or y #TRUE

#The AND operation can also be written as
x & y
#While the OR operation can also be written as
x | y


#MATH FUNCTION AND ITS OPERATIONS             
#importing the math library for performing various mathematical functions (note that we can also use numpy 
#library instead of math library as it's much more generalised and has a wider application
                    
import math #[This has to be ran from console located down right]
x=2
math.log(x) #Gives us the log of the variable
math.sin(x) #Gives us the Sin of the variable

#Not of x
not x
#However, the following would give an error as this isn't possible in Python
!x     

#STRING VALUES AND OPERATIONS
#just like R, both single quotes and double quotes can be used for representing the same
x='Mumbai'
y="Bangalore"
x,y

#Finding the length of a variable in Python
len(x) #would give 6
len(y) #would give 9
x='Mumbai '
len(x) #This would give 7 as it includes the sapce

#We can also try some special characters for our reference
x='Mumbai \t\n'
len(x) #This would give 9, even after appearing to give 11 as output as \t and \n are special characters here

#We can also return a tuple by calling multiple function seperated by a comma in a single line
x='Mumbai'
y="Bangalore"
len(x),len(y)
type((len(x),len(y))) #This would give us a tuple


#String appending
x+y
y+x
#We can also append other static strings, like for e.g.
x+"abc "+y+' xyz'
#The following would give an error
x-y #As we cannot deduct a string from another string

#Upper and lower case conversion (NOTE that of your string contains any special chars, they won't be converted)
x.upper()
x.lower()
#IF we want to capitalize the first letter of a word, we use the following function
x='mumbai'
x.capitalize()

#Aligning or justifying the character
#for right alignment
x="Mumbai"
x.rjust(30) #This means that the text will be right justified by 24 characters, 24 chars coz there are already
            #6 characters in our string and the rjust function will append the rest of the characters so as
            #to appear to a total number of 30 characters as mentioned in the function.
            #NOTE: if the number of characters in our string is more than or equal to what we mentioned in the
            #rjust function, then nothing is going to happen to our texts

#for left alignment
x.ljust(30) #The same number of chars would be appended from the right side this time
            
#for centre alignment
x.center(7) #This is where the spacing needs to be given dependeing upon the no. of chars we have in our string
            #that is here as we have 6 chars for Mumbai, if we say x.center(7), we would get a single spacing at
            #the beginning like ' mumbai', however, if we do x.center(8), we would get something like ' mumbai '
            #Thus it's important to keep in mind the no. of chars in your string var before centre aligning them

x.center(8) #This would give a more appropriate output

#If we want to increase the number of chars by more amt. in our string, we may do so in the following ways
x.center(20)
#or
x.center(50)

#Stripping off spaces or special chars from our string
#spaces from left side
x="             Mumbai"
x.lstrip()
#right strip
x="Mumbai             "
x.rstrip()
#Stripping from both ends
x="            Mumbai              "
x.strip()
#We can also strip special functions if our strings contain the same using lstrip,rstrip and strip functions
x=" Mumbai   \n"
x.strip()
x.lstrip()
x.rstrip()

#We can use the print function for printing a variable
x='Mumbai'
print(x)

#We can use the replace function for replacing a string
x='Mumbai'
x.replace("Mu","Bo") #First include what part of substring to replace and then text that will replcae the 
                     #aforesaid part
                     
                     
#Splitting strings based on char postion, e.g.
x=" Mumbai is a great city "                     
x.split("a") #There will be 4 substrings from this, the way it works is that wherever it will  get an "a" in the
             #main string, it will split the main string from there


#Counting the total number of times a substring occurs in a string
#e.g.
z="Bombay is Bo"             
z.count("Bo") #This will return 2
#Please note that the above operation is case sensitive             
#That means if we try and search "bo" instead of "Bo" in the above scenario, then we would not get desired
#results
#e.g.
z.count("bo") #This will return 0
#In case, we want to check for all occurences of a particular string, we can do something like the following
z.lower().count("bo") #This would return 2
#So by converting the char string to upper or lower case, we can achieve the given req.              


#We can also find the index of the first occurance of a sub-string inside a string
#e.g.
z.find("B") #We would get a 0 here since the first occurance of "B" is at index 0
#Note that just like count function, even find function is case sensitive


#Comparing strings with the double equal to operator
x+y == y+x #Note that this will be false as MumbaiBangalore is not equal to BangaloreMumbai


#Other comparison operators and their operations (NOTE: All of them Return a boolean result)
x=34
y=32

#Comparison "=="
x==y 

#Comparison not equal to "!="
x!=y 

#Comparison greater than
x>y

#Comparison lesser than
x<y

#Comparison greater than equal to
x>=y

#Comparison lesser than equal to
x<=y


#Comparing occurance of strings (This returns a boolean result)
#NOTE that this operation is also applicable to other data structures like  list, tuple etc
x="as"
y="assignment"
x in y 

#Comparison for non-occurance (This is used to check if a sub-string is non occurning in a string or not)
#(This also returns a boolean result)
x not in y


#Note that comparison operations like <,> etc also works with strings. these comparison between strings,
#give result based on dictionary order as defined in the local OS
#(It basically depends on the alphabatical order of the strings and length of the string)
#for e.g.
x<y
#or
y<x
#or
x<=y
#0r
x>=y
#The above comparisons would work even if x and y are strings based on the aforesaid logic 


#LISTS
#List is basically a heterogeneous data structure that can contain items of various types
#It is also an ordered collection
x=[1,2,3,70,-10,0,99,"a","b","c"]
x

#List subsetting just like string subsetting
#e.g.
x[3:7]
#and other subsetting operations would be pretty much same to the one's shown below for string subsetting

#We may also do a step wise subsetting like a loop counter by increasing the count dynamically, this is known
#as step
#e.g.
x[3::2] #This will begin at the the index 3 and end at the end of the string while incrementing 2 objects/chars
        #at a time.
        
#We can also do something like
x[3:-3:2] #This will begin at index 3 and end at the 3rd index from the back, while incrementing 2 objects/chars       
          #at a time.

#We can also try something like the following
x[::-1]  #This will simply take all the objects/chars involved in your list and iterate in reverse order          
#or          
x[::-2] #In this case the iteration would happen with 2 chars and would print the entire list in reverse

#We can also directly refer to the objects of the list directly through index in the following way
#e.g.
x[2]

x[1]

x[9]

x[20] #This would give a list out of range error as there is no object at the 20th index of the list

x[-1]

x[-3]

x[-5]

#Reversing the string
x.reverse() #Note that this would not only reverse an instance of the string, but would change the value
            #of x
            
#So if we try and print x now, we would find that the values of list x have been reversed        
x            
 
#Following is an example of mutability operation: that is we can change the value of any objects of a string
#by referring to them directly
#e.g.
x[2]="Abhi"
x

#We can also do the same thing like the following
x[2:4]="b" #here b will be assigned at multiple indexes, however the way it works is that it will merge the
           #the values in the index 2 and 3 into one index and then add b to it's value, all other indexes 
x          #are adjusted accordingly


#We are now trying to replace the objects of a list with an array of strings
#e.g.
x[2:4]=["abc","def","xyz"]
#So what the above code will do is that it would start apending the array of strings from index 2 to index 
#(4-1) that is index 3 and it would add "abc","def" and "xyz" to the list while shifting the list one place
#right
 
#If we want to append a new object at the end of the list, we can do soething like 
x.append("new")
x
 
#If we wish to append a new list inside a list, we do the following
x.append([1,2,3])
 
#In case you wanna unravel the individual elements in a list and then append them to an existing list,
#We suggest you do the following
x.extend([1,2,3]) #This would extract out the elements of the list to append and then append them to the
x                   #existing list
                   
                   
