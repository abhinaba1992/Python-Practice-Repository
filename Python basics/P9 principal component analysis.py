# -*- coding: utf-8 -*-
"""
Created on Sat May 26 11:01:26 2018

@author: Abhinaba
"""

#The following piece of code represents the Principal component analysis
#and factor analysis (for PCA to work, all the variables must be of type numeric)

#Outlining the Steps for PCA
#1. Data Cleaning
#2. Calculate the correlation matrix
#3. Scaling of data
#4. Initiating the PCA object (Where we need to specify the number of components)
#5. Running the prediction for the PCA components
#6. Checking out the Components and the variable values explained by each component
#7. Checking the explained variance ratio
#8. Seeing the cumulative sum of the variance ratio in order to conclude the optimal number of components
#   we need to use
#9. Initiating the PCA object again (With the new number of components which we would get by seeining the cumulative
#    sum)
#10. Running the prediction for the PCA components
#11. Apply dimensionality reduction transformation
#12. getting a dot product of the transformed values with the correlation matrix
#13. Checking out the Components and the variable values explained by each component again.
#14. Appending the findings to the main data frame



#Importing all the required libraries
import pandas as pd
import math
import numpy as np
import numpy as np
from sklearn.decomposition import PCA
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import scale
get_ipython().magic('matplotlib inline')


#Note that thais file isn't present in the directory
data_file=r'C:\Users\Abhinaba\Desktop\Edvancer Materials\Python\DataCash_Assistance_Engagement_Report.csv'
#data_file2=r'C:\Users\Abhinaba\Desktop\Edvancer Materials\Python\Data\loans data.csv' (Need to do the entire
#data transformation so as to convert all the values to numeric so as to use it for PCA)


#Reading the file
cash=pd.read_csv(data_file)

#Looking at the data
cash.head()
#There are about 66 columns

cash.dtypes
#Checking the data types of all the variables

cash=cash.drop(["Month"],1)
#Dropping the month variable as it's the only variable with the object data type

#We need to first calculate the correlation matrix
cash.corr(method='pearson', min_periods=1)

#We can then check out the data frame or the object again so as to check the 65x65 correlation matrix
#We would be able to see the postive or negative correation of the variables
cash

#Dropping the NAs (Note that the NAs are dropped after calculating correlation because the way correaltion 
#is calculated, it would not affect even if there are NAs in our data so we can choose to drop the NAs either 
#before or after we have calculated correlation, but before we do PCA on our data, the NAs need to be dropped.
cash.dropna(axis=0,inplace=True)


#WE are now taking a copy of that data set
X=cash.copy()


#We are now scaling our data here ((x-mu)/sigma)
X = scale(X)


#Looking at the first few records of X
X[:,:5]


#Checking out the shape of the data
X.shape
#This would give us 6x65 (i.e. 6 rows and 65 columns for the given data set)

#We initialise the PCA object where n_components=4, we basically want to reduce the 65 components or features
#into 4 components
pca = PCA(n_components=4)


#We are doing a PCA.fit to get the components
pca.fit(X)


#We are now checking the components
pca.components_ #this would not make much sense to us now until we see the shape

#We are now checking the shape of the components
pca.components_.shape
#This would basically give us a shape of 4x65 (note that here 4 is not the no. of rows and neither is 65 the 
#number of columns, rather 4 is the new set of columns we have got through PCA and each of this 4 columns
#are represented by the original 65 columns NOTE: the fundamental way PCA works is by calculating the linear
#combination of the existing column set and it gives such a  combination for which the individual PCA components
#are orthogonal to each other.)

#Checking the ratio of the explained variance 
var= pca.explained_variance_ratio_


#Printing the same
print(var)
#Explained variance wwould give us 4 values since we choose 4 components (it would basically give us how much 
#variability of the data is captured by the 4 components with %age distribution (that is if we add these 4 ratios
#we would be getting a value closer to 1), note that the order of the ratios would be in decreasing)


#We are now checking the cumulative sum of the 4 ratios we got in order to understand the total % variation
#captured by our data 
var1=np.cumsum(np.round(pca.explained_variance_ratio_, decimals=4)*100)

print(var1)
#Printing the cumulative sum. We would find out that 90% of variance is captured by first 3 components itself. 
#Now, the task of choosing the number of components is a call of analyst or business people (but by industry
#standard, we consider 90-95% of variance in the original data should be explained by the components.) So, the 
#optimal number of components that explain the variability in our data should be considered as the number of 
#components to be selected by our PCA algo

#Plotting the cumulative sum in the graph
plt.plot(var1)



#Trying diff. number of components (6 components)
#Initiating the PCA object
pca = PCA(n_components=6)


#Running the prediction on the same
pca.fit(X)


#Apply dimensionality reduction to X.
X1=pca.transform(X)


#Taking out the correlation matrix
pd.DataFrame(X1).corr(method='pearson', min_periods=1)


#Getting the components in a variable
loadings=pca.components_

#Seeing the components 
print(*zip(cash.columns,loadings[1,]))

#Note that when better interpretation is the need, then we may not go with PCA because of the fact it tries to
#merge the variability of multiple features into a single component which may not be easy to explain or interpret
#However, when only accuracy or prediction is the scenario at hand, we can go for PCA as it gives us immense
#improve in our performance




#Doing a factor analysis for dimension increment

#importing/loading the libraries required for factor analysis
from sklearn.decomposition import FactorAnalysis
import matplotlib.pyplot as plt


#Setting the path for the file 
data_file='~/Dropbox/March onwards/Python Data Science/Data/cars.csv'

#Reading the file 
cars=pd.read_csv(data_file)

#Checking out the shape for the same
cars.shape

#Checking some of the values
cars.head()

#Dropping the name column as it's an object column
X_cars=cars.drop(['Name'],1)

#Checking the shape of the data frame
X_cars.shape

#Performing scaling on the data set
X_cars=scale(X_cars)


#So, we are performing the factor analysis here based on the given number of components (NOTE that factor analysis
#needs us to give a maximum iteration value as well coz unlike the linear approach which is a linear aljebra driven
#approach, so there is no approximation used. However, Factor analysis is an iterative way of solving the issue and
#it doesn't have a closed form solution coz the VARIMAX algorithm that goes on behind this doesn't have a closed 
#form solution
fa=FactorAnalysis(n_components=4,max_iter=1000)


#We are now fitting our data onto this
fa.fit(X_cars)


#We are now seeing the loading components
loadings=fa.components_


#We are seeing the loading components
#Seeing the first loading
print(*zip(cars.columns[1:],loadings[0,]))

#Seeing the Second loading
print(*zip(cars.columns[1:],loadings[1,]))

#Seeing the Third loading
print(*zip(cars.columns[1:],loadings[2,]))

#Seeing the Fourth loading
print(*zip(cars.columns[1:],loadings[3,]))


#We are now checking the noise variance
nvar=fa.noise_variance_
plt.plot(nvar)
#We would get a line plot where we would have the column indexes in the x-axis while we would be having the noise
#variance values
#So from the above graph, we can conclude that ideally the variables/features for which we get high noise variance
#should ideally be dropped from the data frame before doing factor analysis. So, we would need to do the factor 
#analysis again after dropping the variables with high noise.
