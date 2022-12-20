#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
import warnings
warnings.filterwarnings("ignore")


# In[2]:


df=pd.read_csv("Downloads//brain_stroke.csv")


# In[3]:


df


# In[4]:


df.head()


# In[5]:


df.info()


# In[6]:


df.shape


# In[7]:


df.describe()


# In[8]:


df.tail()


# In[9]:


df.nunique()


# In[10]:


df_ = df.drop(['age','avg_glucose_level','bmi'],axis=1)
for i in df_ .columns:
    print(df_[i].nunique())


# In[11]:


for i in df_ .columns:
    print(df_[i].value_counts())


# In[12]:


import seaborn as sns
for i in df_ .columns:
    plt.figure(figsize=(15,6))
    sns.countplot(df_ [i],data=df_ )


# In[15]:


for i in df_ .columns:
    plt.figure(figsize=(15,6))
    df_ [i].value_counts().plot(kind='pie',autopct='%1.1f%%')


# In[16]:


for i in df_ .columns:
    plt.figure(figsize=(15,6))
    sns.countplot(df_ [i],data=df_ ,hue='stroke')


# In[18]:


#Ratio of people diagnosed with stroke based on Gender




Gender=df.groupby(['gender'])['stroke'].sum()
Gender.plot(kind='pie',autopct='%1.1f%%')

#The shows that female are more diagnosed with stroke with 56.5% than male of 43.5%


# In[21]:


#People who has stroke based on type of work they do.Private,Government,Self-employed and children
worktype=df.groupby(['work_type'])['stroke'].sum()
worktype.plot(kind='pie',autopct='%1.1f%%')

#People who work in private seconds have increase proportion of being diagnosed with stroke (59.7%)


# In[22]:


#Proportion based on Residence
Residence=df.groupby(['Residence_type'])['stroke'].sum()
Residence.plot(kind='pie',autopct='%1.1f%%')

#This shows that people residing in Urabn area have high proportion of being diagnosed with stroke


# In[23]:


#Smoking status proportion

Smoke=df.groupby(['smoking_status'])['stroke'].sum()
Smoke.plot(kind='pie',autopct='%1.1f%%')


# In[25]:


#Dummoes Variable
#df=df.get_dummies(df,columns=['work_type','Resident_type','smoking_status'])
#Instead of using dummies,I will use label encoder

from sklearn.preprocessing import LabelEncoder
encoder=LabelEncoder()
df['gender']=encoder.fit_transform(df['gender'])
df['Residence_type']=encoder.fit_transform(df['Residence_type'])
df['work_type']=encoder.fit_transform(df['work_type'])
df['smoking_status']=encoder.fit_transform(df['smoking_status'])
df['ever_married']=encoder.fit_transform(df['ever_married'])


# In[26]:


pd.DataFrame(df.head())


# In[27]:


#correlation between the varibles

corr=df.corr()
sns.heatmap(corr,annot=True)


# In[29]:


#Training the model

#split the data into dependent and independent variable
X=df.drop(['stroke'],axis=1)
Y=df['stroke']


# In[30]:


#Training and Testing data

from sklearn.model_selection import train_test_split
X_train,X_test,Y_train,Y_test=train_test_split(X,Y,random_state=0,test_size=0.2)

print(X_train.shape)
print(Y_train.shape)
print(X_test.shape)
print(Y_test.shape)



# In[31]:


from sklearn.linear_model import LogisticRegression
lr=LogisticRegression()
lr.fit(X_train,Y_train)


# In[32]:


Y_pred=lr.predict(X_test)
pd.DataFrame(Y_pred)


# In[33]:


from sklearn.metrics import confusion_matrix
cm=confusion_matrix(Y_test,Y_pred)
cm


# In[36]:


#Accuracy

print('Training accuracy:' ,lr.score(X_train,Y_train))
print('Training accuracy:' ,lr.score(X_test,Y_test))


# In[38]:


from sklearn.tree import DecisionTreeClassifier
dtc=DecisionTreeClassifier()
dtc.fit(X_train,Y_train)


# In[39]:


dtc_pred=dtc.predict(X_test)
pd.DataFrame(dtc_pred)


# In[40]:


cm=confusion_matrix(Y_test,dtc_pred)
cm


# In[41]:


print('Training accuracy: ',dtc.score(X_train,Y_train))
print('Training accuracy: ',dtc.score(X_test,Y_test))


# In[42]:


get_ipython().system('apt-get install texlive texlive-xetex texlive-latex-extra pandoc')
get_ipython().system('pip install pypandoc')


# In[ ]:




