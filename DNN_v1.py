
# coding: utf-8

# In[ ]:


import numpy as np
import sys
import tensorflow as tf
from keras import models
from keras.layers import Dense,Dropout,normalization
from keras import optimizers


# In[ ]:


inputfile = './'
with open(inputfile,'r') as f:
    lines = f.readlines()
x_list = []
y_list = []
title = lines.pop(0)
for line in lines:
    data_piece = line.strip().split()
    label = int(data_piece.pop(-1))
    x_list.append(data_piece)
    y_list.append(label)


# In[ ]:


def split_data(x_list,y_list,ratio):
    y_arr = np.array(y_list)
    x_train = []
    x_test = []
    y_train = []
    y_test = []
    for i in range(1,7):
        inds = np.where(y_arr == i)[0]
        train_num = int(len(inds)*ratio)
        train_ind = np.random.choice(inds,train_num,False)
        for ind in list(inds):
            if ind in train_ind:
                x_train.append(x_list[ind])
                y_train.append(i-1)
            else:
                x_test.append(x_list[ind])
                y_test.append(i-1)
    x_train = np.array(x_train).astype(np.float32)
    x_test = np.array(x_test).astype(np.float32)
    y_train = np.array(y_train).astype(np.int32)
    y_test = np.array(y_test).astype(np.int32)  
    return x_train, x_test, y_train, y_test


# In[ ]:


x_train, x_test, y_train, y_test = split_data(x_list,y_list,0.75)
model = models.Sequential()
model.add(Dense(5000,activation='relu'))
model.add(Dense(512,activation='relu'))
model.add(Dense(100,activation='relu'))
model.add(Dense(32,activation='relu'))
model.add(Dense(6,activation='softmax'))
sgd = optimizers.SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)
model.compile(loss='categorical_crossentropy', optimizer=sgd)
y_train = keras.utils.to_categorical(y_train, num_classes=6)
y_test = keras.utils.to_categorical(y_test, num_classes=6)
model.fit(x=x_train, y=y_train, batch_size=5, epochs=10, verbose=1)
loss, acc = model.evaluate(x=x_test, y=y_test)
print('Test accuracy is {:.4f}'.format(acc))

