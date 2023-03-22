

# Importing the Dependencies

# In[1]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import sklearn.datasets
from sklearn.model_selection import train_test_split
from xgboost import XGBRegressor
from sklearn import metrics


# Importing the Boston House Price Dataset

# In[2]:


house_price_dataset = sklearn.datasets.load_boston()


# In[3]:


print(house_price_dataset)


# In[7]:


# Loading the dataset to a Pandas DataFrame
house_price_dataframe = pd.DataFrame(house_price_dataset.data, columns = house_price_dataset.feature_names)


# In[8]:


# Print First 5 rows of our DataFrame
house_price_dataframe.head()


# In[9]:


# add the target (price) column to the DataFrame
house_price_dataframe['price'] = house_price_dataset.target


# In[10]:


house_price_dataframe.head()


# In[11]:


# checking the number of rows and Columns in the data frame
house_price_dataframe.shape


# In[12]:


# check for missing values
house_price_dataframe.isnull().sum()


# In[13]:


# statistical measures of the dataset
house_price_dataframe.describe()


# Understanding the correlation between various features in the dataset

# 1. Positive Correlation
# 
# 2. Negative Correlation

# In[14]:


correlation = house_price_dataframe.corr()


# In[15]:


# constructing a heatmap to nderstand the correlation
plt.figure(figsize=(10,10))
sns.heatmap(correlation, cbar=True, square=True, fmt='.1f', annot=True, annot_kws={'size':8}, cmap='Blues')


# Splitting the data and Target

# In[16]:


X = house_price_dataframe.drop(['price'], axis=1)
Y = house_price_dataframe['price']


# In[17]:


print(X)
print(Y)


# Splitting the data into Training data and Test data

# In[18]:


X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size = 0.2, random_state = 2)


# In[19]:


print(X.shape, X_train.shape, X_test.shape)


# Model Training

# XGBoost Regressor

# In[20]:


# loading the model
model = XGBRegressor()


# In[21]:


# training the model with X_train
model.fit(X_train, Y_train)


# Evaluation

# Prediction on training data

# In[22]:


# accuracy for prediction on training data
training_data_prediction = model.predict(X_train)


# In[23]:


print(training_data_prediction)


# In[24]:


# R squared error
score_1 = metrics.r2_score(Y_train, training_data_prediction)

# Mean Absolute Error
score_2 = metrics.mean_absolute_error(Y_train, training_data_prediction)

print("R squared error : ", score_1)
print('Mean Absolute Error : ', score_2)


# Visualizing the actual Prices and predicted prices

# In[27]:


plt.scatter(Y_train, training_data_prediction)
plt.xlabel("Actual Prices")
plt.ylabel("Predicted Prices")
plt.title("Actual Price vs Preicted Price")
plt.show()


# Prediction on Test Data 

# In[25]:


# accuracy for prediction on test data
test_data_prediction = model.predict(X_test)


# In[26]:


# R squared error
score_1 = metrics.r2_score(Y_test, test_data_prediction)

# Mean Absolute Error
score_2 = metrics.mean_absolute_error(Y_test, test_data_prediction)

print("R squared error : ", score_1)
print('Mean Absolute Error : ', score_2)


# In[ ]:




