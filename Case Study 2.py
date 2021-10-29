#!/usr/bin/env python
# coding: utf-8

# In[78]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


# In[4]:


df = pd.read_csv('casestudy.csv',index_col=0)


# In[5]:


df


# In[8]:


df.year.unique()


# In[44]:


df_2015 = df[df['year']==2015]


# In[10]:


df_2016 = df[df['year']==2016].merge(df[df['year']==2015],how='left',on='customer_email')


# In[17]:


df_2017 = df[df['year']==2017].merge(df[df['year']==2016],how='left',on='customer_email')


# In[26]:


df_2016.columns = ['customer_email','net_revenue_curr','year_curr','net_revenue_prev','year_prev']
df_2017.columns = ['customer_email','net_revenue_curr','year_curr','net_revenue_prev','year_prev']


# In[65]:


def show_res(year):

    df_new = df[df['year']==year].merge(df[df['year']==year-1],how='left',on='customer_email')
    df_new.columns = ['customer_email','net_revenue_curr','year_curr','net_revenue_prev','year_prev']
   
    df_lost = df[df['year']==year-1].merge(df[df['year']==year],how='left',on='customer_email')
    df_lost.columns = ['customer_email','net_revenue_curr','year_curr','net_revenue_prev','year_prev']
    
    # Total revenue for the current year
    ttl_rev = df_new.net_revenue_curr.sum()

    # New Customer Revenue 
    new_rev = df_new[df_new.isnull().T.any()].net_revenue_curr.sum()

    # Existing Customer Growth
    exist_rev_growth = df_new[df_new.notnull().T.all()].net_revenue_curr.sum()-df_new[df_new.notnull().T.all()].net_revenue_prev.sum()

    # Revenue lost from attrition
    lost_rev = df_lost[df_lost.isnull().T.any()].net_revenue_curr.sum()

    # Existing Customer Revenue Current Year
    exist_rev_curr = df_new[df_new.notnull().T.all()].net_revenue_curr.sum()

    # Existing Customer Revenue Prior Year
    exist_rev_prev = df_new[df_new.notnull().T.all()].net_revenue_prev.sum()

    # Total Customers Current Year
    ttl_cstmr_curr = len(df[df['year']==year]['customer_email'].unique())

    # Total Customers Previous Year
    ttl_cstmr_prev = len(df[df['year']==year]['customer_email'].unique())

    # New Customers
    new_cstmr = len(df_new[df_new.isnull().T.any()])

    # Lost Customers
    lost_cstmr = len(df_lost[df_lost.isnull().T.any()])


    res = pd.DataFrame.from_dict({'total_revenue':[ttl_rev],
                                      'new_customer_revenue':[new_rev],
                                      'existing_customer_growth':[exist_rev_growth],
                                      'revenue_lost':[lost_rev],
                                      'existing_revenue_curr':[exist_rev_curr],
                                      'existing_revenue_prior':[exist_rev_prev],
                                      'total_customer_curr':[ttl_cstmr_curr],
                                      'total_customer_prior':[ttl_cstmr_prev],
                                      'new_customer':[new_cstmr],
                                      'lost_customer':[lost_cstmr]})
    return res


# In[70]:


res_2016 = show_res(2016)


# In[71]:


res_2017 = show_res(2017)


# In[76]:


df_2015 = df[df['year']==2015]
plt.hist(df_2015['net_revenue'])


# In[81]:


sns.boxplot(x='year',y='net_revenue',data = df, showfliers=False)
plt.show()


# In[85]:


df[df['year']==2015]


# In[87]:


[df[df['year']==2015]['net_revenue'].sum(),res_2016['total_revenue'],res_2017['total_revenue']]


# In[99]:


import random
r = random.random()
b = random.random()
g = random.random()
color=(r, g, b)

plt.bar(['2015','2016','2017'],[df[df['year']==2015]['net_revenue'].sum(),res_2016['total_revenue'],res_2017['total_revenue']],color=color,width=0.5)
plt.title('total revenue')


# In[102]:


[res_2016['new_customer_revenue'],res_2017['new_customer_revenue']]


# In[116]:


r = random.random()
b = random.random()
g = random.random()
color=(r, g, b)


plt.bar(['2016','2017'],[res_2016['new_customer_revenue'][0],res_2017['new_customer_revenue'][0]],color=color,width=0.5)
plt.title('new customer revenue')


# In[111]:


r = random.random()
b = random.random()
g = random.random()
color=(r, g, b)

plt.bar(['2016','2017'],[res_2016['existing_customer_growth'][0],res_2017['existing_customer_growth'][0]],color=color,width=0.5)
plt.title('existing customer_growth')


# In[ ]:




