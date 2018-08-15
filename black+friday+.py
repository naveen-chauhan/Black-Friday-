
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
get_ipython().magic('matplotlib inline')


# In[2]:


train=pd.read_csv(r'C:\Users\naveen chauhan\Desktop\mldata\mlp\black friday\train.csv')


# In[3]:


train.shape


# In[4]:


train.head()


# In[5]:


train.isnull().sum()


# In[6]:


train.Age.hist(bins=50)


# In[7]:


train.Occupation.value_counts()


# In[8]:


test=pd.read_csv(r'C:\Users\naveen chauhan\Desktop\mldata\mlp\black friday\test.csv')


# In[9]:


test.shape


# In[10]:


test.isnull().sum()


# In[11]:


train.dtypes


# In[12]:


train.City_Category.value_counts()


# In[13]:


train.Stay_In_Current_City_Years.value_counts()


# In[14]:


train.Age.value_counts()


# In[15]:


train.Product_Category_3.value_counts()


# In[16]:


train.Product_ID.value_counts()


# In[17]:


target=train.Purchase


# In[18]:


train.drop('Purchase',axis=1,inplace=True)


# In[19]:


train.head()


# In[20]:


data=train.append(test)


# In[21]:


data.shape


# In[22]:


data.Product_Category_2.fillna(0,inplace=True)


# In[23]:


data.Product_Category_3.fillna(0,inplace=True)


# In[24]:


data.isnull().sum()


# In[25]:


data['No_Of_Category']=0


# In[26]:


data.head()


# In[27]:


data['No_Of_Category']=np.where(data['Product_Category_2']>float(0), 1, 0)


# In[28]:


data['No_Of_Category']+=np.where(data['Product_Category_3']>float(0), 1, 0)


# In[29]:


data['No_Of_Category']+=1


# In[30]:


data.head()


# In[31]:


data.drop(['Product_Category_2','Product_Category_3'],inplace=True,axis=1)


# In[32]:


data.head()


# In[33]:


from sklearn.preprocessing import LabelEncoder


# In[34]:


le=LabelEncoder()


# In[35]:


data.Product_ID=le.fit_transform(data.Product_ID)


# In[36]:


data.head()


# In[37]:


data.Gender=data.Gender.map({'M':1,'F':0})


# In[38]:


data.Age=le.fit_transform(data.Age)


# In[39]:


data.head()


# In[40]:


data.City_Category=data.City_Category.map({'A':0,'B':1,'C':2})


# In[41]:


data.Stay_In_Current_City_Years=le.fit_transform(data.Stay_In_Current_City_Years)


# In[42]:


data.dtypes


# In[43]:


# we train and make prediction 
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestRegressor
#lasso regressionn ,svm ,multivariate algorthm


# In[44]:


train=data.iloc[:550068,]


# In[45]:


test=data.iloc[550068:,]


# In[46]:


train.shape


# In[47]:


target.shape


# In[48]:


train_X,test_X,train_y,test_y=train_test_split(train,target,random_state=42)


# In[49]:


train_X.shape


# In[50]:


test_X.shape


# In[51]:


from sklearn.linear_model import LinearRegression
from sklearn.svm import SVR
from sklearn.linear_model import Lasso,Ridge
from sklearn.metrics import mean_squared_error
from math import sqrt


# In[52]:


clf=LinearRegression()
clf.fit(train_X,train_y)
pred=clf.predict(test_X)
print(sqrt(mean_squared_error(test_y,pred)))


# In[53]:


clf4=RandomForestRegressor()
clf4.fit(train_X,train_y)
pred4=clf4.predict(test_X)
print(sqrt(mean_squared_error(test_y,pred4)))


# In[ ]:


clf3=Ridge()
clf3.fit(train_X,train_y)
pred3=clf3.predict(test_X)
print(sqrt(mean_squared_error(test_y,pred3)))


# In[ ]:


clf2=SVR()
clf2.fit(train_X,train_y)
pred2=clf2.predict(test_X)
print(sqrt(mean_squared_error(test_y,pred2)))


# In[ ]:


clf1=Lasso()
clf1.fit(train_X,train_y)
pred1=clf.predict(test_X)
print(sqrt(mean_squared_error(test_y,pred1)))

