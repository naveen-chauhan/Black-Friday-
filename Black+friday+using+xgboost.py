
# coding: utf-8

# In[1]:


import numpy as np
import matplotlib.pyplot as plt
get_ipython().magic('matplotlib inline')
import xgboost as xgb


# In[2]:


import pandas as pd
import sys


# In[3]:


train=pd.read_csv(r'C:\Users\naveen chauhan\Desktop\mldata\mlp\black friday\train.csv')


# In[4]:


train.head()


# In[5]:


test=pd.read_csv(r'C:\Users\naveen chauhan\Desktop\mldata\mlp\black friday\test.csv')


# In[6]:


train.Purchase.hist(bins=50)


# In[7]:


cutoff_purchase=np.percentile(train.Purchase,99.9)


# In[8]:


test_user_id=test.User_ID.copy()
print(test_user_id.shape)


# In[9]:


test_product_id=test.Product_ID.copy()


# In[10]:


test_product_id.shape


# In[11]:


train.loc[train.Purchase>cutoff_purchase,'Purchase']=cutoff_purchase


# In[12]:


train.Purchase.hist(bins=50)


# In[13]:


from sklearn.preprocessing import LabelEncoder
le=LabelEncoder()


# In[14]:


train.User_ID=le.fit_transform(train.User_ID)
test.User_ID=le.transform(test.User_ID)


# In[15]:


new_product_id=list(set(np.unique(test.Product_ID))-set(np.unique(train.Product_ID)))


# In[16]:


print(new_product_id)


# In[17]:


le=LabelEncoder()
train.Product_ID=le.fit_transform(train.Product_ID)


# In[18]:


test.loc[test.Product_ID.isin(new_product_id),'Product_ID']=-1
new_product_id.append(-1)


# In[19]:


test.loc[~test.Product_ID.isin(new_product_id),'Product_ID']=le.transform(test.loc[~test.Product_ID.isin(new_product_id),'Product_ID'])


# In[20]:


y=train.Purchase


# In[21]:


train.drop(['Purchase','Product_Category_2','Product_Category_3'],inplace=True,axis=1)


# In[22]:


test.drop(['Product_Category_2','Product_Category_3'],inplace=True,axis=1)


# In[23]:


train.head()


# In[24]:


test.head()


# In[25]:


train=pd.get_dummies(train)


# In[26]:


train.head()


# In[27]:


test=pd.get_dummies(test)


# In[28]:


test.head()


# In[29]:


#now we use xgboost for regression purpos and compute rmse
#Modelling
dtrain=xgb.DMatrix(train.values,label=y,missing=np.nan)


# In[30]:


param={'onjective':'reg:linear','booster':'gbtree','silent':0,'max_depth':10,'eta':0.1,'subsample':0.8
       ,'colsample_bytree':0.8,'min_child_weight':20,'max_delta_step':0,'gamma':0}


# In[31]:


num_round=690


# In[34]:


seed= [1122, 2244, 3366, 4488, 5500]


# In[36]:


test_preds = np.zeros((len(test), len(seed)))


# In[37]:


#has taken approx 1 hr
for run in range(len(seed)):
    sys.stdout.write("\rXGB RUN:{}/{}".format(run+1, len(seed)))
    sys.stdout.flush()
    param['seed'] = seed[run]
    clf = xgb.train(param, dtrain, num_round)
    dtest = xgb.DMatrix(test.values,missing=np.nan)
    test_preds[:, run] = clf.predict(dtest)


# In[38]:


test_preds = np.mean(test_preds, axis=1)


# In[41]:


submit = pd.DataFrame({'User_ID': test_user_id, 'Product_ID': test_product_id, 'Purchase': test_preds})
submit = submit[['User_ID', 'Product_ID', 'Purchase']]

submit.loc[submit['Purchase'] < 0, 'Purchase'] = 12  # changing min prediction to min value in train
submit.to_csv(r"C:\Users\naveen chauhan\Desktop\mldata\mlp\black friday\final_solution.csv", index=False)

