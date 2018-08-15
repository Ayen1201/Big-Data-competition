
# coding: utf-8

# In[49]:


import keras
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten, Activation
from keras import optimizers


# In[50]:


import numpy as np
import pandas as pd


# In[51]:


Xdata_1 = np.zeros((30,1,30000))

for x in range(0,29):
    Xdata_1[x] = pd.read_excel('{num}.xls'.format(num = x)).drop('Col5',axis=1).values.reshape((1,-1))
    
X_train = Xdata_1.reshape(30,30000)


# In[52]:


X_train.shape


# In[53]:


Xdata_2 = np.zeros((10,1,30000))

for x in range(30,39):
    Xdata_2[x-30] = pd.read_excel('{num}.xls'.format(num = x)).drop('Col5',axis=1).values.reshape((1,-1))
    
X_test = Xdata_2.reshape(10,30000)


# In[54]:


X_test.shape


# In[55]:


Ydata = np.array([0.490, 0.306, 0.418, 0.504, 0.499, 0.848, 0.654, 0.473, 0.453, 0.399, 
                  0.551, 0.425, 0.588, 0.747, 0.443, 0.324, 0.571, 0.667, 0.554, 0.705,
                  0.926, 0.492, 0.715, 0.647, 0.626, 0.743, 1.110, 1.073, 0.684, 0.347,
                  0.636, 0.331, 0.574, 0.473, 0.370, 0.563, 0.845, 0.928, 0.418, 0.404]).reshape(-1,1)

# In[56]:


Y_train = Ydata[0:30]
Y_test = Ydata[30:40]


# In[57]:


Y_train.shape


# In[58]:


Y_test.shape


# In[59]:


model = Sequential()
model.add(Dense(input_dim=30000, units=1, activation='tanh'))
model.add(Dropout(0.2))
model.compile(loss='mse', optimizer='sgd')


# In[60]:


model.fit(X_train, Y_train, batch_size=1, epochs=100, initial_epoch=0)


# In[61]:


score = model.evaluate(X_test, Y_test, batch_size=1)
test_data = model.predict(X_test, batch_size=1)
print (test_data)
print (score)

