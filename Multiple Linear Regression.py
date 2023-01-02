#!/usr/bin/env python
# coding: utf-8

# # Covid-19 data 

# ## Importing the relevent libraries

# In[1]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# ## Importing the covid dataset

# In[2]:


dataset = pd.read_csv("covid_data.csv")


# In[3]:


dataset.head()


# In[4]:


X = dataset.iloc[:, :-1].values
y = dataset.iloc[:, -1].values


# ## Importing Scikit Learn Library for categorical data

# In[5]:


from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder


# ### Encoding the categorical variables

# In[6]:


ct = ColumnTransformer(transformers = [("encoder", OneHotEncoder(), [1])], remainder = "passthrough")
X = np.array(ct.fit_transform(X))


# In[7]:


print(X)


# ## Training and Testing the dataset

# ### Dividing the data in train and test data

# In[8]:


from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.4, random_state = 0)


# ## Using Linear Regression

# In[9]:


from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(X_train, y_train)


# ### Reshaping of the data

# In[10]:


# predicting the test set results
y_pred = regressor.predict(X_test)
np.set_printoptions(precision = 3)
print(np.concatenate((y_pred.reshape(len(y_pred), 1), y_test.reshape(len(y_test), 1)), 1))


# ## Plotting the Graph for predicted and test data

# In[11]:


# Visualizing results
plt.plot(y_pred)
plt.plot(y_test)
plt.show()


# ## Determining difference between Test and Predicted data

# In[12]:


# y_pred = regressor.predict(X_test)

print(str(y_test) + " - " + str(y_pred))


# In[13]:


regressor_preds = regressor.predict(X_test)


# ## Finding accuracy of the trained data using test data

# In[14]:


regressor.fit(X_train, y_train)
accuracies = {}
accuracy = regressor.score(X_test, y_test)*100

accuracies["Linear Regression"] = accuracy
print("Test Accuracy {:.2f}%".format(accuracy))


# # We have got a trained ML model with accuracy of 96.92%
