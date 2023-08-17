#!/usr/bin/env python
# coding: utf-8

# In[17]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score


# In[18]:


data = pd.read_csv('Salary_Data.csv')

# Display the first few rows of the dataset
print(data.head())


# In[19]:


X = data['YearsExperience'].values.reshape(-1, 1)
y = data['Salary'].values

# print(X)


# In[20]:


from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


# In[21]:


model = LinearRegression()
model.fit(X_train, y_train)


# In[22]:


y_pred = model.predict(X_test)


# In[23]:


mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print("Mean Squared Error:", mse)
print("R-squared:", r2)


# In[24]:


plt.scatter(X_test, y_test, color='blue', label='Actual Data')
plt.plot(X_test, y_pred, color='red', linewidth=2, label='Linear Regression')
plt.xlabel('Years of Experience')
plt.ylabel('Salary')
plt.legend()
plt.title('Linear Regression: Salary Prediction')
plt.show()


# In[ ]:




