#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd 
import numpy as np
import matplotlib.pyplot as plt
import time


# In[2]:


num= [20,21,22,23]
df = [pd.read_csv('C:\\Users\\default.DESKTOP-2ISHQBS\\lab\\task1_lee\\data\\demand{}.csv'.format(num[i]), encoding='cp949') for i in range(len(num))]
train_set = pd.concat([df[0],df[1]],axis=0)
vaild_set = df[2]
test_set=df[3]

train_set.shape, vaild_set.shape,test_set.shape


# In[3]:


class sample:
    #####################################################################################
    ### 데이터프레임, 날짜 입력
    ### 
    def __init__(self,df,str):
        self.df = df
        self.date= df[str]
        self.date=pd.to_datetime(self.date)
    def monthly(self):    
    #####################################################################################
    ### 시간별 데이터 -> 월별 데이터로
    ### 인덱스 : 년/월, 변수 : y        
        self.train = pd.DataFrame({'date':self.date,'y':self.df.iloc[:,1:].sum(axis=1).values})
        self.train = self.train.groupby(self.train['date'].dt.to_period('M')).sum(numeric_only=True)
        #self.index= self.df.index
        #self.sample = pd.DataFrame(self.df.sum(axis=1))
        #self.sample.columns = ['y']
        return self.train


# In[4]:


target_train =sample(train_set,'날짜')
target_vaild =sample(vaild_set,'날짜')
target_test =sample(test_set,'날짜')


# In[5]:


class WINdow:
    def __init__(self,df,timestep):
        self.df = df
        self.timestep=timestep+1 # 예상한 timestep보다 1적기 때문에 +1
        
    def window(self):
        for i in range(1, self.timestep):
            df['shift_{}'.format(i)] = df.iloc[:,0].shift(i)
            df['shift_{}'.format(i)] = df.iloc[:,0].shift(i)
        window_df = df.dropna(axis=0) # 결측치 공간 제거
        self.window_df = window_df.iloc[:,::-1] # 좌우 반전
        
                
        self.feature= self.window_df.iloc[:,:-1].values
        self.y_label= self.window_df.iloc[:,-1].values
        
        return self. window_df


# In[6]:


df = target_train.monthly()
df_val = target_vaild.monthly()
df_test = target_test.monthly()


# In[9]:


from sklearn.preprocessing import MinMaxScaler
minmax = MinMaxScaler()


# In[10]:


target=WINdow(df,18)
target_scale = minmax.fit_transform(target.window().iloc[:,:-1]/10000)

target_X = target_scale[:,:12]
target_y = target_scale[:,12:]



target_X.shape, target_y.shape 


# In[11]:


target_scale_X = target_X # minmax.fit_transform(target_X)
target_scale_y =target_y #minmax.fit_transform(target_y)


# In[17]:


target_ele_X_train= target_scale_X[:-18,:] 
target_ele_y_train= target_scale_y[:-18,:] 

target_ele_X_test=target_scale_X[[-1],:] 
target_ele_y_test=target_scale_y[[-1],:] 


# In[ ]:




