# -*- coding: utf-8 -*-
"""
Created on Sat Dec 15 21:12:42 2018

@author: Rebecca
"""

import os
from sklearn.model_selection import KFold
os.chdir('C:/Users/Rebecca/Desktop/machinelearning/all')
import numpy as np
import pandas as pd
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import seaborn as sns
import datetime
#from sklearn import cross_validation
from sklearn.preprocessing import LabelEncoder
from airbnbbookingfunctions import *
from pcaf import *
from sklearn.neural_network import MLPClassifier
# Importing the Data :

train_users = pd.read_csv('C:/Users/Rebecca/Desktop/machinelearning/all/train_users_2.csv')
test_users = pd.read_csv('C:/Users/Rebecca/Desktop/machinelearning/all/test_users.csv')
# sessions = pd.read_csv('/Users/quentinpradelle/Desktop/EssecCentrale/Cours Centrale/Machine Learning/AssignmentsAndProject/GroupProject/DataSets/sessions.csv')
countries = pd.read_csv('C:/Users/Rebecca/Desktop/machinelearning/all/countries.csv')
age_gender = pd.read_csv('C:/Users/Rebecca/Desktop/machinelearning/all/age_gender_bkts.csv')

train_labels = train_users['country_destination'] 

# Merging train and test users so we can clean it.

users = pd.concat((train_users, test_users), axis=0, ignore_index=True)


# Remove ID's and set it as index 

users.set_index('id', inplace = True)

#_______________________________________________DATA CLEANING ________________________________________________

# HANDLING MISSING VALUES :

# There are two types of missing values: NA and -unknown-
# We will first transform -unknown- values in NA values

users.replace('-unknown-', np.nan, inplace=True) 

# Here, we will check where the missing values appear 

users_nan = (users.isnull().sum() / users.shape[0]) * 100
users_nan[users_nan > 0].drop('country_destination')


# The two most important features to treat in missing values are gender and age.

# -- Gender -- :

users.gender.value_counts(dropna = False).plot(kind='bar', color='#FD5C64')
plt.xlabel('Gender')


# There is a lot of nan values + "other" values to treat. Not a big 
# difference between male and females. Let's see if we can fill the nan with 
# the country destination:

women = sum(users['gender'] == 'FEMALE')
men = sum(users['gender'] == 'MALE')
female_destinations = users.loc[users['gender'] == 'FEMALE', 'country_destination'].value_counts() / women * 100 # Percentage of Women for each destination
male_destinations = users.loc[users['gender'] == 'MALE', 'country_destination'].value_counts() / men * 100 # Percentage of Men for each destination
width = 0.4 # Bar width
male_destinations.plot(kind='bar', width=width, color='#4DD3C9', position=0, label='Male', rot=0)
female_destinations.plot(kind='bar', width=width, color='#FFA35D', position=1, label='Female', rot=0)
plt.legend()
plt.xlabel('Destination Country')
plt.ylabel('Percentage')
sns.despine()
plt.show()

# We will set the Na values of gender to 'Non defined', and set gender as a dummy variable later on

users['gender'][users.gender.isnull()] = 'Non Defined'


# -- Age -- :

users[users.age > 85]['age'].describe()
users[users.age < 18]['age'].describe()

# There is a lot of users with an age > 85 yo or < 18 yo. We will transform these values in nan:

users['age'][users.age > 85] = np.nan 

# Now, as a wrong or a missing age might be an important information, we will transform
# it in -1 values.

users['age'][users.age.isnull()] = -1

#age is the only feature for which it makes sense to check for outliers
import seaborn as sns
sns.boxplot(x=users['age']) #no outliers


bins = [-1, 18, 24, 29, 34,39,44,49,54,59,64,69,74,79,85]
users['age_group'] = np.digitize(users['age'], bins, right=True)
users.drop(['age'], axis = 1, inplace = True)



# -- First browser --

users.first_browser.value_counts(dropna=False)

# We can see that Chrome, Safari, Firefox and IE are the most used browsers (with also na values).
# We will transform the rest in "other browsers". Then, we will transform this variable in dummy feature.


users['first_browser'][(users.first_browser != 'Chrome') & (users.first_browser != 'Safari') &
                      (users.first_browser != 'Firefox') & (users.first_browser != 'Mobile Safari') &
                      (users.first_browser != 'IE') & (users.first_browser.notnull()) ] = 'Other browsers'


# We will consider Nan values as non identified browsers. 

users['first_browser'][users.first_browser.isnull()] = 'Non identified browser'


# -- Dates -- :

# First, we will convert the dates to a date format

users['date_account_created'] = pd.to_datetime(users['date_account_created'])
users['date_first_booking'] = pd.to_datetime(users['date_first_booking'])
users['date_first_active'] = pd.to_datetime(users['timestamp_first_active'], format='%Y%m%d%H%M%S')

# We will check the accounts created by month 

df = users[~users['country_destination'].isnull()]
df.groupby([df["date_account_created"].dt.year, df["date_account_created"].dt.month])['country_destination'].count().plot(kind="bar",figsize=(20,10))

# There seem to be a correlation between the month of account creation and first booking.
# We will group months into four seasons, and then in dummy variables

users['month_account_created'] = pd.DatetimeIndex(users['date_account_created']).month
users['season_account_created'] = pd.DatetimeIndex(users['date_account_created']).month

users['season_account_created'][(users.month_account_created == 12) |
                                (users.month_account_created == 1) |
                                (users.month_account_created == 2)] = 'Winter'
users['season_account_created'][(users.month_account_created == 3) |
                                (users.month_account_created == 4) |
                                (users.month_account_created == 5)] = 'Spring'
users['season_account_created'][(users.month_account_created == 6) |
                                (users.month_account_created == 7) |
                                (users.month_account_created == 8)] = 'Summer'
users['season_account_created'][(users.month_account_created == 9) |
                                (users.month_account_created == 10) |
                                (users.month_account_created == 11)] = 'Autumn'

        
# Change the month from int to 'Str' so as to treat it like a categorical variable 

users['month_account_created'] = users['month_account_created'].map({1 : 'January', 2 : 'February', 3 : 'March', 4 : 'April', 5 : 'May', 6 : 'June', 7 : 'July', 8 : 'August', 9 : 'September', 10 : 'October', 11 : 'November', 12 : 'December'})

# Here, we drop the dates and will try our first model without it

users.drop(['date_first_booking'], axis = 1, inplace = True)
users.drop(['date_account_created'], axis = 1, inplace = True)
users.drop(['date_first_active'], axis = 1, inplace = True)


# --- First affiliate tracked --- (First marketing interaction with the user)

users.first_affiliate_tracked.value_counts(dropna  = False)

# We will just convert nan values into 'No marketing done', and later convert this variable to a dummy variable

users['first_affiliate_tracked'][users.first_affiliate_tracked.isnull()] = 'No Marketing done'


# -- Language --

users.language.value_counts(dropna = False)

# There is only one na value, we will transform it to "other"

users['language'][users.language.isnull()] = 'other'

# Normally, there is no missing values anymore:

users_nan = (users.isnull().sum() / users.shape[0]) * 100
users_nan.drop('country_destination')


# _____ DEALING WITH CATEGORICAL VALUES _____

# Now, we want to convert character values to numeric values to build our model. 
# We will convert our features into dummy variables. 

# First, give the type 'category' to category variables:

users.select_dtypes(include=['object'])

categorical_features = [
        'country_destination',
   'affiliate_channel',
   'affiliate_provider',
   'gender',
   'first_affiliate_tracked',
   'first_browser',
   'first_device_type',
   'language',
   'signup_app',
   'signup_method',
   'season_account_created',
   'month_account_created'
]

for categorical_feature in categorical_features:
   users[categorical_features] = users[categorical_features].astype('category')


# Now we create our new user dataframe changing categorical variables to Dummy variables
users_cleaned = pd.get_dummies(users, prefix = 'dummy_', columns = categorical_features, drop_first = True)

users_cleaned.select_dtypes(include=['object'])
list(users_cleaned)
#creating ds_pca_red for second pca
ds_pca_red=users_cleaned.iloc[:,[3,4,5,6,7,8,9,10,11,12,13,15,19,20,21,26,29,45,48,53,57,62,89]]
#users_cleaned['country_destination'][(users_cleaned.country_destination == "NDF")] = 1
#users_cleaned['country_destination'][(users_cleaned.country_destination == "US")] = 2
#users_cleaned['country_destination'][(users_cleaned.country_destination == "other")] = 3
#users_cleaned['country_destination'][(users_cleaned.country_destination == "FR")] = 4
#users_cleaned['country_destination'][(users_cleaned.country_destination == "IT")] = 5
#users_cleaned['country_destination'][(users_cleaned.country_destination == "GB")] = 6
#users_cleaned['country_destination'][(users_cleaned.country_destination == "ES")] = 7
#users_cleaned['country_destination'][(users_cleaned.country_destination == "CA")] = 8
#users_cleaned['country_destination'][(users_cleaned.country_destination == "DE")] = 9
#users_cleaned['country_destination'][(users_cleaned.country_destination == "NL")] = 10
#users_cleaned['country_destination'][(users_cleaned.country_destination == "AU")] = 11
#users_cleaned['country_destination'][(users_cleaned.country_destination == "PT")] = 12

# Check if there are nan values remaining / 418 are normal (test set survived column), 1 is not normal 
users_cleaned.isnull().values.any()
users_cleaned.isnull().sum().sum()

# Get the index of country destination column 
idx_destination = users_cleaned.columns.get_loc("country_destination")

# ________________________________________________MAIN FUNCTION_____________________________________________
list(users_cleaned)
# Transform Dataframe to np array

users_cleaned = users_cleaned.convert_objects(convert_numeric = True)
users_cleaned = users_cleaned.values

 
#pca1
dataset=users_cleaned[:213451, :]
from pcaf import *
array=pcaf(dataset) 


#pcared with significant variables
ds_pca_red = ds_pca_red.convert_objects(convert_numeric = True)
ds_pca_red = ds_pca_red.values
from pcared import *
array=pcared(dataset) 
 

#scale data
users_cleaned=ScaleData(users_cleaned)
#Separating user_cleaned into train and test sets

X_train = users_cleaned[:213451, :]
X_test = users_cleaned[213451:, :]
Y_train = X_train[:, idx_destination] 

print(Y_train)

# Now we drop the column 'country_destination' that will be the one we try to predict

X_train = delete(X_train, 0, 1)
X_test = delete(X_test, 0, 1)

#X_train = delete(X_train, range(79,97), 1)
#X_test = delete(X_test, range(79,97), 1)

-------------------------------------#MODELS----------------------------------------------
# Initialize cross validation
kf = cross_validation.KFold(X_train.shape[0], n_folds=10)

totalInstances = 0 # Variable that will store the total intances that will be tested  
totalCorrect = 0 # Variable that will store the correctly predicted intances  

# Loop for testing the model 

for trainIndex, testIndex in kf:
    trainSet = X_train[trainIndex]
    testSet = X_train[testIndex]
    trainLabels = Y_train[trainIndex]
    testLabels = Y_train[testIndex]
	
    predictedLabels = DecisionTree(trainSet, trainLabels, testSet)

    correct = 0	
    for i in range(testSet.shape[0]):
        if predictedLabels[i] == testLabels[i]:
            correct += 1
    
    print 'Accuracy: ' + str(float(correct)/(testLabels.size))
    totalCorrect += correct
    totalInstances += testLabels.size
print 'Total Accuracy: ' + str(totalCorrect/float(totalInstances))

print(predictedLabels)



--------------------#let's try to add a NEURAL NETWORK and see if accuracy improves--------------------
#neural network
x_train,x_test,y_train,y_test=train_test_split(X_train,Y_train,test_size=0.25,random_state=123)
x_train=ScaleData(x_train)
clf=MLPClassifier()
cv_score2=cross_val_score(estimator=clf,X=x_train,y=y_train,cv=10)
print(cv_score2) #max accuracy 0.5843

#again changing parameters
nn=MLPClassifier(hidden_layer_sizes=(200, ),activation='relu',solver='sgd')
x_train=ScaleData(x_train)
cv_score3=cross_val_score(estimator=nn,X=x_train,y=y_train,cv=10)
print(cv_score3) #best score 0.5843

nn=MLPClassifier(hidden_layer_sizes=(300, ),activation='relu',solver='sgd')
x_train=ScaleData(x_train)
cv_score4=cross_val_score(estimator=nn,X=x_train,y=y_train,cv=10)
print(cv_score4) #best score 0.5843 again




------------------#LOG-REG-Tuned




#for log-reg-tuned we have to change things a bit cause it takes too much time
#try with a smaller dataset
users_cleaned=users_cleaned.sample(n=20000)
X_test=users_cleaned.loc[users_cleaned['country_destination'].isnull()]
X_train=users_cleaned.loc[users_cleaned['country_destination'].notnull()]

# Transform Dataframe to np array

X_test = X_test.convert_objects(convert_numeric = True)
X_test = X_test.values 

X_train = X_train.convert_objects(convert_numeric = True)
X_train = X_train.values 

# Separating user_cleaned into train and test sets

#X_train = users_cleaned[:213451, :]
#X_test = users_cleaned[213451:, :]
Y_train = X_train[:, idx_destination] 

print(Y_train)

# Now we drop the column 'country_destination' that will be the one we try to predict

X_train = delete(X_train, 0, 1)
X_test = delete(X_test, 0, 1)

#X_train = delete(X_train, range(79,97), 1)
#X_test = delete(X_test, range(79,97), 1)
#

#AC##################################################################################
totalInstances = 0 # Variable that will store the total intances that will be tested  
totalCorrect = 0 # Variable that will store the correctly predicted intances  


predictedLabels = LogRegTuned(X_train, Y_train,X_test)
#AC##################################################################################