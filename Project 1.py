#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
#loading the csv data to pandas
heart_data=pd.read_csv('C:\\Users\\deept\\Downloads\\heart_disease_data.csv')
#print first 5 rows of dataset
heart_data.head()
#print last 5 rows
heart_data.tail()
heart_data.info()


# In[3]:


heart_data.isnull().sum()


# In[5]:


#statistical measure about the data
heart_data.describe()


# In[6]:


heart_data['target'].value_counts()


# In[9]:


#splitting the features and target
X=heart_data.drop(columns='target',axis=1)
Y=heart_data['target']
print(X)


# In[10]:


print(Y)


# In[11]:


X_train,X_test,Y_train,Y_test=train_test_split(X,Y,test_size=0.2,stratify=Y,random_state=2)


# In[12]:


print(X.shape,X_train.shape,X_test.shape)


# In[13]:


#Logistic Regression 
model = LogisticRegression()
# training the logisticRegression model with Training data
model.fit(X_train,Y_train)


# In[14]:


#accuracy on training data
X_train_prediction = model.predict(X_train)
training_data_accuracy=accuracy_score(X_train_prediction,Y_train)


# In[15]:


print("Accuracy on Training Data :",training_data_accuracy)


# In[16]:


#accuracy on test data
X_test_prediction = model.predict(X_test)
test_data_accuracy=accuracy_score(X_test_prediction,Y_test)
print("Accuracy On Test Data :",test_data_accuracy)


# In[15]:


#Building A Predictive System


# In[17]:


input_data = (60,1,0,117,230,1,1,116,1,1.4,2,2,3)

#change input data to a numpy array
input_data_as_numpy_array=np.asarray(input_data)

# reshape the numpy array as we are predeicting for only one instance
input_data_reshaped = input_data_as_numpy_array.reshape(1,-1)
prediction = model.predict(input_data_reshaped)
print(prediction)
if(prediction[0]==0):
    print('The Person does not have a Heart Disease ')
else:
    print('The person has Heart Disease')


# In[ ]:





# In[ ]:




