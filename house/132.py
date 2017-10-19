# -*- coding: utf-8 -*-
"""
Created on Fri Oct  6 15:45:31 2017

@author: en
"""

import pandas as pd
import numpy as np


#read dataset into X and Y
df=pd.read_csv('train-v3.csv')
dataset=df.values
X_train=dataset[:,2:22]
y_train=dataset[:, 1]

df2=pd.read_csv('test-v3.csv')
dataset2=df2.values 
X_test=dataset2[:,1:21]

df3=pd.read_csv('valid-v3.csv')
dataset3=df3.values
X_valid=dataset3[:, 2:22]
y_valid=dataset3[:, 1]

#
def normalize(train,valid,test):

    tmp=train
    mean,std=tmp.mean(axis=0),tmp.std(axis=0)
    print("tmp.shape=",tmp.shape)
    print("mean.shape=",mean.shape)
    print("std.shape=",std.shape)
    print("mean=",mean)
    print("std=",std)
    train= (train-mean) / std
    valid= (valid-mean) / std
    test= (test-mean) / std
    return train, valid, test
    
    
from keras.models import Sequential
from keras.layers.core import Dense, Activation
from keras.layers import Dropout

X_train,X_valid,X_test=normalize(X_train,X_valid,X_test)

#neural network

model=Sequential()

#model.add(Reshape((4,5,1),input_shape=(n,)))

model.add(Dense(32,input_dim=20,init='normal', activation='relu'))
model.add(Dropout(0,1))
model.add(Dense(120,input_dim=32,init='normal', activation='relu'))
model.add(Dropout(0,2))
model.add(Dense(160,input_dim=120,init='normal', activation='relu'))
model.add(Dropout(0,2))
model.add(Dense(160,input_dim=160,init='normal', activation='relu'))
model.add(Dropout(0,2))
model.add(Dense(120,input_dim=160,init='normal', activation='relu'))
model.add(Dropout(0,2))
model.add(Dense(20,input_dim=120,init='normal', activation='relu'))
model.add(Dropout(0,2))

model.add(Dense(1,init='normal'))

#complie model
model.compile(loss='mae',optimizer='adam')

#train&test
model.fit(X_train, y_train, epochs=170, batch_size=32,verbose=1,validation_data=(X_valid, y_valid))


#pred=model.predict(X_test)

Y_predict=model.predict(X_test)
np.savetxt('testt.csv', Y_predict, delimiter=',')




