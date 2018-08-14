
# coding: utf-8

# In[1]:


import keras
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten, Activation
from keras import optimizers


# In[2]:


import numpy as np
import pandas as pd


# In[3]:


Xtrain = np.zeros((30,1,30000))

for x in range(0,29):
    Xtrain[x] = pd.read_excel('{num}.xls'.format(num = x)).drop('Col5',axis=1).values.reshape((1,-1))
    


# In[4]:


Xtest = np.zeros((10,1,30000))

for x in range(30,39):
    Xtest[x-30] = pd.read_excel('{num}.xls'.format(num = x)).drop('Col5',axis=1).values.reshape((1,-1))


# In[7]:


Ydata = np.array([0.49, 0.306, 0.418, 0.504, 0.499, 0.848, 0.654, 0.473, 0.453, 0.399, 0.551
                 , 0.425, 0.588, 0.747, 0.443, 0.324, 0.571, 0.667, 0.554, 0.705, 0.9264
                 ,0.492, 0.7149, 0.647, 0.626, 0.743, 1.11, 1.0736, 0.684, 0.347, 0.636
                 , 0.331, 0.574, 0.473, 0.370, 0.563, 0.845, 0.928, 0.418, 0.404]).reshape(-1,1)


# In[14]:


X_train = np.vstack((Xtrain[0],Xtrain[1]))

for i in range (2,30):
    X_train = np.vstack((X_train, Xtrain[i])) 

X_test = np.vstack((Xtest[0],Xtest[1]))

for i in range (2,10):
    X_test = np.vstack((X_test, Xtest[i]))


# In[15]:


Y_train = Ydata[0:30]
Y_test = Ydata[30:40]


# In[16]:


X_train.shape


# In[17]:


Y_train.shape


# In[19]:


X_test.shape


# In[23]:


Y_test.shape


# In[20]:


model = Sequential()
model.add(Dense(input_dim=30000, units=1, activation='tanh'))
model.add(Dropout(0.2))
model.compile(loss='mse', optimizer='sgd')


# In[21]:


model.fit(X_train, Y_train, batch_size=1, epochs=100, initial_epoch=0)


# In[22]:


score = model.evaluate(X_test, Y_test, batch_size=5)
test_data = model.predict(X_test, batch_size=1)
print (test_data)
print (score)


# In[24]:

