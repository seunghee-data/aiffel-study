#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import seaborn as sns


# In[2]:


car_df = pd.read_csv('/aiffel/aiffel/cars/cars.csv')
brand_df = pd.read_csv('/aiffel/aiffel/cars/brand.csv')


# In[3]:


car_df.head()


# In[4]:


brand_df.head()


# In[5]:


car_df['brand'] = car_df['title'].str.split().str[0]


# In[6]:


car_df.head()


# In[7]:


brand_df['title'] = brand_df['title'].str.upper()


# In[8]:


brand_df.head()


# In[9]:


car_df = car_df.merge(brand_df, left_on='brand', right_on='title', how='left')


# In[10]:


car_df.head()


# In[11]:


car_df = car_df.drop(columns=['title_y'])


# In[12]:


car_df = car_df.rename(columns={'title_x': 'title'})


# In[13]:


car_df.head()


# In[14]:


bonus_df = car_df.copy()


# In[15]:


car_df.info()


# In[16]:


car_df.head()


# In[17]:


bonus_df['Engine'] = bonus_df['Engine'].astype(str).str.replace('L', '', regex=False)


# In[18]:


bonus_df['Emission Class'] = bonus_df['Emission Class'].str.extract('(\d+)')


# In[19]:


bonus_df['Engine'] = pd.to_numeric(bonus_df['Engine'], errors='coerce')
bonus_df['Emission Class'] = pd.to_numeric(bonus_df['Emission Class'], errors='coerce')
car_df['Engine'] = bonus_df['Engine']


# In[20]:


car_df.info()


# In[21]:


car_df.describe()


# In[22]:


car_df.isna().mean()


# In[23]:


car_df['Service history'].unique()


# In[24]:


bonus_df.groupby('Service history')['Price'].mean()


# In[25]:


bonus_df['Service history'] = bonus_df['Service history'].fillna('Unknown')


# In[26]:


car_df.groupby('Service history')['Price'].mean()


# In[27]:


car_df[car_df['Engine'].isna()]


# In[28]:


car_df['na_values'] = car_df.isna().sum(axis = 1)


# In[29]:


car_df.head()


# In[30]:


len(car_df[car_df['na_values'] >= 4])


# In[31]:


car_df = car_df[car_df['na_values'] < 4]


# In[32]:


car_df.drop('na_values', axis = 1, inplace = True)


# In[33]:


car_df.isna().mean()


# In[34]:


sns.displot(car_df['Previous Owners'])


# In[35]:


car_df['Previous Owners'].median()


# In[36]:


sns.displot(car_df['Engine'])


# In[37]:


car_df['Engine'].mean()


# In[38]:


car_df['Engine'].median()


# In[39]:


sns.displot(car_df['Doors'])


# In[40]:


car_df['Doors'].median()


# In[41]:


sns.displot(car_df['Seats'])


# In[42]:


sns.displot(car_df['Emission Class'])


# In[43]:


car_df = car_df.fillna(car_df.select_dtypes(include='number').median())


# In[44]:


car_df.describe()


# In[45]:


car_df['Price'].sort_values()


# In[46]:


car_df['Mileage(miles)'].sort_values()


# In[47]:


car_df[car_df['Mileage(miles)'] < 1000]


# In[48]:


car_df = car_df[car_df['Mileage(miles)'] > 1000]


# In[49]:


car_df['Registration_Year'].sort_values()


# In[50]:


car_df = car_df[car_df['Registration_Year'] < 2025]


# In[51]:


car_df['Previous Owners'].sort_values()


# In[52]:


car_df[car_df['Previous Owners'] == 9]


# In[53]:


car_df.groupby('brand')['Price'].agg(['mean', 'std'])


# In[54]:


car_df.pivot_table(values='Price', index='brand', columns='Fuel type')


# In[55]:


car_df.head()


# In[56]:


sns.scatterplot( x= car_df['Previous Owners'], y = car_df['Price'])


# In[57]:


sns.scatterplot( x= car_df['Registration_Year'], y = car_df['Price'])


# In[58]:


sns.scatterplot( x= car_df['Registration_Year'], y = np.log(car_df['Price']))


# In[59]:


car_df.head()


# In[60]:


car_df[['title','Fuel type','Body type','Gearbox','Emission Class','Service history','brand','country']].nunique()


# In[61]:


car_df.drop('title', axis = 1, inplace = True)


# In[62]:


car_df['brand'].value_counts()


# In[63]:


car_df.groupby('brand')['Price'].mean()


# In[64]:


bonus_df['Engine'] = bonus_df['Engine'].astype(str).str.replace('L', '', regex=False)


# In[65]:


car_df = pd.get_dummies(car_df, drop_first = True)


# In[67]:


from sklearn.preprocessing import RobustScaler


# In[68]:


rs = RobustScaler()


# In[69]:


car_df = pd.DataFrame(rs.fit_transform(car_df), columns=car_df.columns)


# In[70]:


from sklearn.decomposition import PCA


# In[71]:


from sklearn.decomposition import PCA
pca = PCA(n_components=5)


# In[72]:


car_df_pca = pca.fit_transform(car_df)


# In[73]:


pca.explained_variance_ratio_


# In[74]:


for i in range(2, 11):
    pca = PCA(i)
    pca.fit(car_df)
    print(i, round(pca.explained_variance_ratio_.sum(), 2))


# In[75]:


pca = PCA(7)


# In[76]:


PCA(n_components=7).fit_transform(car_df)


# In[78]:


bonus_df.groupby('country')['brand'].nunique()


# In[79]:


bonus_df.corr()


# In[ ]:




