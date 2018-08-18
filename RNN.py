
# coding: utf-8

# In[ ]:


import keras
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten, Activation, CuDNNGRU
from keras import optimizers


# In[ ]:


import numpy as np
import pandas as pd


# In[ ]:


X_train = np.zeros((30, 7500, 4))

for x in range(0,29):
    X_train[x] = pd.read_excel('{num}.xls'.format(num = x)).drop('Col5',axis=1).values


# In[ ]:


X_train = X_train*10000000


# In[ ]:


X_test = np.zeros((10,7500,4))

for x in range(30,39):
    X_test[x-30] = pd.read_excel('{num}.xls'.format(num = x)).drop('Col5',axis=1).values
X_test = X_test*10000000


# In[ ]:


X_test


# In[ ]:


Ydata = np.array([0.490, 0.306, 0.418, 0.504, 0.499, 0.848, 0.654, 0.473, 0.453, 0.399, 
                  0.551, 0.425, 0.588, 0.747, 0.443, 0.324, 0.571, 0.667, 0.554, 0.705,
                  0.926, 0.492, 0.715, 0.647, 0.626, 0.743, 1.110, 1.073, 0.684, 0.347,
                  0.636, 0.331, 0.574, 0.473, 0.370, 0.563, 0.845, 0.928, 0.418, 0.404]).reshape(-1,1)


# In[ ]:


Y_train = Ydata[0:30]
Y_test = Ydata[30:40]


# In[ ]:


Y_train.shape


# In[ ]:


Y_test.shape


# In[ ]:


model = Sequential()
model.add(CuDNNGRU(50,
    batch_input_shape=(1,7500,4),
    return_sequences=False,
    stateful=True
))

# In[ ]:

model.add(Dense(100))
model.add(Dense(1))
adam = keras.optimizers.Adam(lr=0.0001, beta_1=0.9, beta_2=0.999, epsilon=None, decay=0.0, amsgrad=False)
model.compile(optimizer=adam,loss='mse')


# In[ ]:


model.fit(X_train, Y_train, epochs=100, batch_size=1)

# In[ ]:


test_data = model.predict(X_test, batch_size=1)
print ("test_data = \n",test_data,'\n\n')
diff = abs(test_data-Y_test)
print ("Different = \n",diff,'\n')


# In[ ]:


diff_sqr = diff**2
RMSE =( diff_sqr.sum()/10 )**0.5
print("RMSE = ",RMSE)




