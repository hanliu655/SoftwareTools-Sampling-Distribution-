#!/usr/bin/env python
# coding: utf-8

# In[8]:


import numpy as np
import pandas as pd
from sympy import *
import matplotlib.pyplot as plt


# In[3]:


# Study the change in error rate for Gamma distribution


# In[10]:


data=pd.read_csv("RMSE_gamma.csv")
data.head()


# In[12]:


class LinearRegression:
    
    def __init__(self,data,y,degree=1):
        self.data=data
        self.y=y
        self.degree=degree

        #Creat the DesignMatrix can fit differnt degrees       
    def DesignMatrix(self):
        df_x=self.data.drop(self.y,1)
        
        for col in df_x.columns:
            for deg in range(2, self.degree + 1):
                df_x[f"{col}^{deg}"] = df_x[col] ** deg
        df_x.insert(0,'beta',1)
        X=Matrix(df_x.to_numpy())
        self.X=X
        
    def ResponseMatrix(self):
        df_y=self.data[self.y]
        Y=Matrix(df_y.to_numpy())
        self.Y=Y
    
    def beta_values(self):
        self.DesignMatrix()
        self.ResponseMatrix()
        return (((self.X.T*self.X).inv())*(self.X.T)*self.Y)
    
    def y_hat(self):
        self.DesignMatrix()
        self.ResponseMatrix()
        return self.X@(((self.X.T*self.X).inv())*(self.X.T)*self.Y)
    
    def Residuals(self):
        self.DesignMatrix()
        self.ResponseMatrix()
        return numpy.subtract(self.Y, self.X@(((self.X.T*self.X).inv())*(self.X.T)*self.Y))
    
    def norm_Residuals(self):
        self.DesignMatrix()
        self.ResponseMatrix()
        return (self.Y - self.X@(((self.X.T*self.X).inv())*(self.X.T)*self.Y)).norm()
    
    def y_vs_y_hat_plot(self):
        self.DesignMatrix()
        self.ResponseMatrix()
        return plt.scatter(self.Y,(self.X@(((self.X.T*self.X).inv())*(self.X.T)*self.Y)))


# In[13]:


# Calculate norm values for degrees 1-14
norms = []
for x in range(1,15):
    lr=LinearRegression(data,y='RMSE',degree=x)
    norm=lr.norm_Residuals()
    norms.append(norm)
    print(norms)


# In[14]:


min_norm = min(norms)
print(f"The minimum norm of residuals is {min_norm}")


# In[15]:


# plot norm of residuals for polynonmials versus different degrees 


# In[16]:


plt.plot(range(1, 15), norms)
plt.xlabel('Degree')
plt.ylabel('Norm of residuals')
plt.title('Norm of residuals for polynomial fits of different degrees')
plt.show()


# In[36]:


#Fits a polynomial equation with degree of 14
import matplotlib.pyplot as plt
import numpy as np


# In[18]:


lr = LinearRegression(data, y='RMSE', degree=14) 
beta = lr.beta_values() 


# In[19]:


y_hat=lr.y_hat() 


# In[39]:


# plot RMSE with best fitted line 


# In[ ]:





# In[21]:


X_plot = np.linspace(data['n'].min(), data['n'].max(), 500)


# In[22]:


y_hat.shape


# In[23]:


X_plot.shape


# In[30]:


plt.scatter(data['n'], data['RMSE'])
plt.plot(np.array(X_plot), np.array(y_hat), color='red')
plt.xlabel('Sample Sizd')
plt.ylabel('RMSE for Gamma Distribution')
plt.title('RMSE with fitted line')
plt.show()


# In[ ]:





# In[42]:


# 1. Simulate f'(x) = (f(x + delta_x) - f(x - delta_x)) / 2 * delta_x


# In[25]:


delta_x = 0.01 
derivative1 = [(y_hat[i+1] - y_hat[i-1]) / (2 * delta_x) 
                for i in range(1, len(y_hat)-1)]
print(derivative1)


# In[26]:


sum(derivative1)


# In[27]:


#2. Simulate f'(x) = (f(x + delta_x) - f(x)) / delta_x


# In[28]:


derivative2 = [(y_hat[i+1] - y_hat[i]) / (delta_x) 
                for i in range(1, len(y_hat)-1)]
print(derivative2)


# In[29]:


sum(derivative2)
# Second derivitive provides a better approximation of the derivitive since dealt x is smaller 


# In[ ]:





# In[ ]:





# In[ ]:




