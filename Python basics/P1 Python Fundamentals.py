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
#e.g. 1
import Keyword #for importing a library named Keyword
#Now if we wish to import specific components from the library, we can do
Keyword.Kwlist

#e.g. 2
#We can also import parts of a library in the following way
from Keyword import Kwlist
#Now the above approach is useful when we do not have another component with the same name in another library, like for e.g. we cannot have
#"from Keyword import Kwlist" and "from <some other library> import Kwlist" in the same piece of code as it will cause problem, so its always
#a good practice to import the library first and then refer to specific portions of the same like shown in example 1

#e.g. 3
from Keyword import *
#This is very much similar to e.g. 1, however, the only difference is that in example 1 you have to refer to specific components by preceding
#them with the name of the library and a dot(e.g. Keyword.Kwlist), but in e.g. 3 you can directly include the components with the component 
#name(Kwlist)

#e.g. 4
import Keyword as kw
kw.Kwlist

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
from sys import sizeof
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
