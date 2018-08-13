
# coding: utf-8

# In[55]:


import keras
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten, Activation
from keras import optimizers


# In[56]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# In[57]:


Xdata = np.zeros((4,7500,4))

for x in range(1,3):
    Xdata[x] = pd.read_excel('{num}.xls'.format(num = x)).drop('Col5',axis=1).values


# In[58]:


Xdata.shape


# In[59]:


Xdata[1].shape


# In[60]:


Ydata1 = np.full((7500,1),0.306)


# In[61]:


Ydata2 = np.full((7500,1),0.418)


# In[62]:


Ydata3 = np.full((7500,1),0.505)


# In[63]:


X_train , Y_train = np.vstack((Xdata[1],Xdata[2])) , np.vstack((Ydata1, Ydata2))


# In[64]:


X_test , Y_test = Xdata[3], Ydata3


# In[65]:


model = Sequential()
model.add(Dense(input_dim=4, units=1))
model.compile(loss='mse', optimizer='sgd')


# In[66]:


model.fit(X_train, Y_train, batch_size=6, epochs=100, initial_epoch=0)


# In[ ]:


score = model.evaluate(X_train, Y_train, batch_size=5)
test_data = model.predict(X_test, batch_size=1)
print (test_data)
print (model.layers[0].get_weights(), '\n')
print (score)

