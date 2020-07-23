#!/usr/bin/env python
# coding: utf-8

# In[20]:


from sklearn import datasets
import pandas as pd
from sklearn import model_selection
from sklearn.linear_model import LinearRegression


# In[16]:


boston=datasets.load_boston()
x=boston.data
y=boston.target


# In[51]:


df=pd.DataFrame(x)
df.columns=boston.feature_names
features=boston.feature_names
df["dis_dis"]=df.DIS**2
x2=df.values


# In[54]:


x_train,x_test,y_train,y_test=model_selection.train_test_split(x,y,random_state=0)
x2_train,x2_test,y2_train,y2_test=model_selection.train_test_split(x2,y,random_state=0)
alg1=LinearRegression()
alg2=LinearRegression()
alg1.fit(x_train,y_train)
alg2.fit(x2_train,y2_train)


# In[56]:


train_score=alg1.score(x_train,y_train)
test_score=alg1.score(x_test,y_test)
print("Train score:",train_score)
print("Test_score:",test_score)


train2_score=alg2.score(x2_train,y2_train)
test2_score=alg2.score(x2_test,y2_test)
print("Train_score 2:",train2_score)
print("Test_score 2:",test2_score)


# In[35]:


features=boston.feature_names

