
# coding: utf-8

# In[13]:


import keras
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten, Activation, CuDNNGRU
from keras.optimizers import adam


# In[2]:


import numpy as np
import pandas as pd


# In[3]:


X_train = np.zeros((30, 7500, 4))

for x in range(0,29):
    X_train[x] = pd.read_excel('{num}.xls'.format(num = x)).drop('Col5',axis=1).values


# In[4]:


X_train.shape


# In[5]:


X_test = np.zeros((10,7500,4))

for x in range(30,39):
    X_test[x-30] = pd.read_excel('{num}.xls'.format(num = x)).drop('Col5',axis=1).values


# In[6]:


X_test.shape


# In[7]:


Ydata = np.array([0.490, 0.306, 0.418, 0.504, 0.499, 0.848, 0.654, 0.473, 0.453, 0.399, 
                  0.551, 0.425, 0.588, 0.747, 0.443, 0.324, 0.571, 0.667, 0.554, 0.705,
                  0.926, 0.492, 0.715, 0.647, 0.626, 0.743, 1.110, 1.073, 0.684, 0.347,
                  0.636, 0.331, 0.574, 0.473, 0.370, 0.563, 0.845, 0.928, 0.418, 0.404]).reshape(-1,1)


# In[8]:


Y_train = Ydata[0:30]
Y_test = Ydata[30:40]


# In[9]:


Y_train.shape


# In[10]:


Y_test.shape


# In[14]:


model = Sequential()
model.add(CuDNNGRU(4,
    input_shape=(7500, 4),
    return_sequences=False,
))


# In[12]:


model.add(Dense(1))
model.compile(optimizer='adam',loss='mse')


# In[13]:


model.fit(X_train, Y_train, epochs=100, batch_size=10)


# In[14]:


test_data = model.predict(X_test, batch_size=1)
print (test_data)

