#!/usr/bin/env python
# coding: utf-8

# In[36]:


import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib import style
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score,recall_score,f1_score


# In[2]:


df = pd.read_csv('Churn_Modelling.csv')


# In[3]:


df.head()


# In[4]:


df.info()


# In[5]:


df.describe()


# In[6]:


df.columns


# In[7]:


df.shape


# In[8]:


#Get encoding of categorical  features


# In[9]:


df['Gender'] = df['Gender'].replace(['Male', 'Female'], [0,1])
df


# In[10]:


df['Geography'] = df['Geography'].replace(['France', 'Spain', 'Germany'], [0,1,2,])
df


# In[11]:


y = df['EstimatedSalary']


# In[12]:


y.shape


# In[13]:


y


# In[14]:


x = df [['RowNumber', 'CustomerId', 'CreditScore', 'Geography',
       'Gender', 'Age', 'Tenure', 'Balance', 'NumOfProducts', 'HasCrCard',
       'IsActiveMember', 'Exited']]


# In[15]:


df=df.drop(['RowNumber', 'CustomerId', 'Surname'], axis=1)


# In[16]:


df.head()


# In[17]:


df['Geography'].unique()


# In[18]:


df['Exited'].value_counts()


# In[ ]:





# In[ ]:





# In[ ]:





# In[19]:


x.shape


# In[20]:


x


# In[21]:


from sklearn.preprocessing import StandardScaler


# In[22]:


sc = StandardScaler()


# In[23]:


x_std = df[['Age', 'Tenure']]


# In[24]:


x_std = sc.fit_transform(x_std)


# In[25]:


x_std 


# In[38]:


x[['Age', 'CustomerId']] = pd.DataFrame(x_std, columns= ['Age','CustomerId'])


# In[27]:


x


# In[28]:


#Get Train Test Split


# In[29]:


x_train, x_test, y_train, y_test = train_test_split(x,y,test_size=0.2,random_state=2529)


# In[30]:


x_train.shape,x_test.shape, y_train.shape,y_test.shape


# In[31]:


#get model train


# In[32]:


from sklearn.ensemble import RandomForestRegressor


# In[33]:


rfr = RandomForestRegressor(random_state=2529)


# In[34]:


df = df.replace(r'^\s*$', np.nan, regex=True)


# In[35]:


rfr.fit(x_train,y_train)


# In[ ]:


# get model prediction


# In[39]:


y_pred = rfr.predict(x_test)


# In[40]:


y_pred.shape


# In[41]:


y_pred


# In[42]:


from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score


# In[43]:


mean_squared_error(y_test,y_pred)


# In[44]:


mean_absolute_error(y_test,y_pred)


# In[45]:


r2_score(y_test,y_pred)


# In[47]:


#Get visualization of actual vs predicted results


# In[49]:


plt.scatter(y_test,y_pred)
plt.xlabel("Actual ")
plt.ylabel("predicted" )
plt.title("Actual  vs predicted ")
plt.show()

