
# coding: utf-8

# In[1]:


import sys
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split


# In[2]:


import sys


# In[3]:


import pandas as pd


# In[4]:


import matplotlib.pyplot as plt


# In[5]:


import seaborn as sns


# In[6]:


import sklearn


# In[7]:


games = pd.read_csv("games.csv")


# In[8]:


print(games.columns
     )


# In[9]:


print(games.shape)


# In[10]:


plt.hist(games["average_rating"])


# In[11]:


print(games[games["average_rating"] == 0].iloc[0])
print(games[games["average_rating"] > 0].iloc[0])


# In[12]:


#Premove rows with user reviews
games = games[games["users_rated"] > 0]

#Remove missing values

games = games.dropna(axis = 0)


# In[13]:


plt.hist(games["average_rating"])
plt.show()


# In[14]:


print(games.columns)


# In[15]:


#Correlation Matrix


# In[16]:


corrmat = games.corr()


# In[17]:


fig = plt.figure(figsize = (12 , 9))
sns.heatmap(corrmat , vmax = 0.8 , square = True)
plt.show()


# In[22]:


columns = games.columns.tolist()
columns = [c for c in columns if c not in ["bayes_average_rating" , "average_rating" , "type" , "name" , "id"]]

#Store the variable we'll be predicting on
target = "average_rating"


# exit()

# In[23]:


columns = games.columns.tolist()
columns = [c for c in columns if c not in ["bayes_average_rating" , "average_rating" , "type" , "name" , "id"]]

#Store the variable we'll be predicting on
target = "average_rating"


# In[24]:


from sklearn.model_selection import train_test_split

train = games.sample(frac = 0.8 , random_state = 1)

#Select anything not in training set

test = games.loc[~games.index.isin(train.index)]

#Print shapes

print(train.shape)


# In[25]:


print(test.shape)


# In[27]:


#Import linear regression model
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

LR = LinearRegression()


# In[28]:


LR.fit(train[columns] , train[target])


# In[30]:


#Generate predictions for the testing set

predictions = LR.predict(test[columns])

# Compute error between our test predictions 

mean_squared_error(predictions , test[target])


# In[33]:


from sklearn.ensemble import RandomForestRegressor

#Initialise the model

RFR = RandomForestRegressor(n_estimators = 100 , min_samples_leaf = 10 , random_state = 1)
RFR.fit(train[columns] , train[target])

from sklearn.ensemble import RandomForestRegressor


#Initialise the model

RFR = RandomForestRegressor(n_estimators = 100 , min_sample_leaf = 10 , random_state = 1)
RFR.fit()
# In[34]:


predictions = RFR.predict(test[columns])

mean_squared_error(predictions , test[target])


# In[35]:


test[columns].iloc[0]


# In[36]:


rating1 = LR.predict(test[columns].iloc[0].values.reshape(1 , -1))

rating2 = RFR.predict(test[columns].iloc[0].values.reshape(1 , -1))

print(rating1)

print(rating2)


# In[37]:


print(test[target].iloc[0])

