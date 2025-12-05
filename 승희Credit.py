#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt


# In[2]:


cc_df = pd.read_csv('fraud.csv')


# In[3]:


pd.set_option('display.max_columns', 50)
pd.set_option('display.max_rows', 100)


# In[5]:


cc_df.info()


# In[6]:


cc_df.describe()


# In[7]:


cc_df.info()


# In[8]:


cc_df.head(3)


# In[9]:


cc_df['merchant'].nunique()


# In[10]:


cc_df['job'].nunique()


# In[11]:


cc_df['cc_num'].nunique()


# In[12]:


cc_df.drop(['merchant','first','last','street','city','state','zip','job','trans_num','unix_time'], axis=1, inplace=True)


# In[13]:


cc_df.sort_values('cc_num')


# In[14]:


temp = pd.DataFrame({'a': [10,20,30,20,10,200], 'b': [100,300,200,150,250,200], 'c': [10,500,20,250,25,200]})


# In[15]:


temp


# In[16]:


temp.mean()


# In[17]:


temp.std()


# In[18]:


(temp['a'] - 48.33) / 74.67


# In[19]:


(temp['b'] - temp['b'].mean()) / temp['b'].std()


# In[20]:


(temp['c'] - temp['c'].mean()) / temp['c'].std()


# In[21]:


cc_df['cc_num'].value_counts()


# In[22]:


amt_info = cc_df.groupby('cc_num')['amt'].agg(['mean','std']).reset_index()


# In[23]:


amt_info.to_pickle('./amt_info.pkl')


# In[24]:


cc_df = cc_df.merge(amt_info, on='cc_num', how='left')


# In[25]:


cc_df['amt_z'] = (cc_df['amt'] - cc_df['mean']) / cc_df['std']


# In[26]:


cc_df[cc_df['is_fraud'] == 1]


# In[27]:


cc_df.drop(['mean','std'], axis=1, inplace=True)


# In[28]:


cat_info = cc_df.groupby(['cc_num','category'])['amt'].agg(['mean','std']).reset_index()


# In[29]:


cat_info.to_pickle('./cat_info.pkl')


# In[30]:


cc_df = cc_df.merge(cat_info, on=['cc_num','category'], how='left')


# In[31]:


cc_df['cat_amt_z'] = (cc_df['amt'] - cc_df['mean']) / cc_df['std']


# In[32]:


cc_df.drop(['mean','std'], axis=1, inplace=True)


# In[33]:


cc_df.head()


# In[34]:


cc_df['hour'] = pd.to_datetime(cc_df['trans_date_trans_time']).dt.hour


# In[35]:


cc_df.head()


# In[36]:


def hour_func(x):
    if (x >= 6) & (x < 12):
        return 'morning'
    elif (x >= 12) & (x < 18):
        return 'afternoon'
    elif (x >= 18) & (x < 23):
        return 'night'
    else:
        return 'evening'


# In[37]:


cc_df['hour_cat'] = cc_df['hour'].apply(hour_func)


# In[38]:


cc_df.head()


# In[39]:


cc_df['hour_cat'].value_counts()


# In[40]:


all_cnt = cc_df.groupby('cc_num')['amt'].count().reset_index()


# In[41]:


hour_cnt = cc_df.groupby(['cc_num','hour_cat'])['amt'].count().reset_index()


# In[42]:


all_cnt.head()


# In[43]:


hour_cnt.head()


# In[44]:


hour_cnt = hour_cnt.merge(all_cnt, on='cc_num', how='left')


# In[45]:


hour_cnt.head()


# In[46]:


hour_cnt = hour_cnt.rename(columns={'amt_x': 'hour_cnt', 'amt_y': 'total_cnt'})


# In[47]:


hour_cnt.head()


# In[48]:


hour_cnt['hour_perc'] = hour_cnt['hour_cnt'] / hour_cnt['total_cnt']


# In[49]:


hour_cnt.head(10)


# In[50]:


cc_df.head()


# In[51]:


hour_cnt = hour_cnt[['cc_num','hour_cat','hour_perc']]


# In[52]:


hour_cnt.to_pickle('./hour_cnt.pkl')


# In[53]:


cc_df = cc_df.merge(hour_cnt, on=['cc_num','hour_cat'], how='left')


# In[54]:


cc_df.head()


# In[55]:


cc_df.drop(['trans_date_trans_time', 'hour', 'hour_cat'], axis=1, inplace=True)


# In[56]:


get_ipython().system('pip install geopy')


# In[57]:


from geopy.distance import distance


# In[58]:


distance((48.8878, -118.2105), (49.159047, -118.186462)).km


# In[59]:


cc_df['distance'] = cc_df.apply(lambda x: distance((x['lat'], x['long']), (x['merch_lat'], x['merch_long'])).km, axis=1)


# In[60]:


from datetime import datetime


# In[61]:


start_time = datetime.now()
cc_df.head(10000).apply(lambda x: distance((x['lat'], x['long']), (x['merch_lat'], x['merch_long'])).km, axis=1)
datetime.now() - start_time


# In[62]:


cc_df.head()


# In[63]:


dist_info = cc_df.groupby('cc_num')['distance'].agg(['mean','std']).reset_index()


# In[64]:


dist_info.to_pickle('./dist_info.pkl')


# In[65]:


cc_df = cc_df.merge(dist_info, on='cc_num', how='left')


# In[66]:


cc_df.head()


# In[67]:


cc_df['dist_z'] = (cc_df['distance'] - cc_df['mean']) / cc_df['std']


# In[68]:


cc_df


# In[69]:


cc_df.drop(['lat','long','merch_lat','merch_long','mean','std'], axis = 1, inplace = True)


# In[71]:


cc_df['dob'] = pd.to_datetime(cc_df['dob']).dt.year


# In[75]:


cc_df['category'].nunique()


# In[77]:


cc_df = pd.get_dummies(cc_df, drop_first=True)


# In[78]:


cc_df.head()


# In[79]:


cc_df.drop('cc_num', axis=1, inplace=True)


# In[ ]:




