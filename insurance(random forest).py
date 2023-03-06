#!/usr/bin/env python
# coding: utf-8

# In[99]:


import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib import style
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


# In[5]:


df = pd.read_csv('insurance.csv')


# In[6]:


df.head()


# In[7]:


df.info()


# In[8]:


df.describe()


# In[12]:


df[['smoker']].value_counts()


# In[15]:


df[['region']].value_counts()


# In[16]:


df[['sex']].value_counts()


# In[17]:


#column names
df.columns


# df.shape

# In[41]:


df['sex'] = df['sex'].replace(['male', 'female'], [0,1])
df


# In[43]:


df['smoker'] = df['smoker'].replace(['no', 'yes'], [0,1])
df 


# In[45]:


df['region'] = df['region'].replace(['southwest', 'southeast', 'northwest','northeast'], [0,1,2,3])
df


# In[50]:


y = df['charges']


# In[51]:


y.shape


# In[52]:


y


# In[66]:


x= df[['age','sex','bmi','children','smoker','region']]


# In[81]:


x= df.drop (['charges'],axis=1)


# In[82]:


x.shape


# In[83]:


x


# In[ ]:





# In[53]:


from sklearn.preprocessing import StandardScaler


# In[54]:


sc = StandardScaler()


# In[56]:


x_std = df[['age', 'bmi']]


# In[58]:


x_std = sc.fit_transform(x_std)


# In[59]:


x_std 


# In[ ]:


x[['age', 'bmi']] = pd.DataFrame(x_std, columns= ['age','bmi'])


# In[84]:


x


# In[ ]:





# In[ ]:


#Get Train Test Split


# In[101]:


x_train, x_test, y_train, y_test = train_test_split(x,y,test_size=0.2,random_state=2529)


# In[103]:


x_train.shape,x_test.shape, y_train.shape,y_test.shape


# In[ ]:


#get model train


# In[105]:


from sklearn.ensemble import RandomForestRegressor


# In[106]:


rfr = RandomForestRegressor(random_state=2529)


# In[107]:


rfr.fit(x_train,y_train)


# In[ ]:


# get model prediction


# In[108]:


y_pred = rfr.predict(x_test)


# In[109]:


y_pred.shape


# In[110]:


y_pred


# In[114]:


from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score


# In[115]:


mean_squared_error(y_test,y_pred)


# In[116]:


mean_absolute_error(y_test,y_pred)


# In[117]:


r2_score(y_test,y_pred)


# In[ ]:


#Get visualization of actual vs predicted results


# In[119]:


plt.scatter(y_test,y_pred)
plt.xlabel("Actual prices")
plt.ylabel("predicted prices")
plt.title("Actual prices vs predicted prices")
plt.show()


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[92]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:




