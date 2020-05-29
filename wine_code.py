#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd

dataset=pd.read_csv('wines.csv')
Y=dataset['Class']
y=pd.get_dummies(Y)
X=dataset.drop('Class',axis=1)

from keras.models import Sequential
from keras.optimizers import RMSprop
from keras.layers import Dense
from keras.backend import clear_session
clear_session()
model=Sequential()

from sklearn.preprocessing import StandardScaler
sc=StandardScaler()

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.35, random_state=50)
X_train=sc.fit_transform(X_train)

model.add(Dense(units=8, input_shape=(13,),activation='relu', kernel_initializer='he_normal'))

model.add(Dense(units=8,activation='relu', kernel_initializer='he_normal'))
model.add(Dense(units=8,activation='relu', kernel_initializer='he_normal'))
model.add(Dense(units=8,activation='relu', kernel_initializer='he_normal'))
model.add(Dense(units=8,activation='relu', kernel_initializer='he_normal'))
model.add(Dense(units=8,activation='relu', kernel_initializer='he_normal'))

model.add(Dense(units=3,activation='softmax'))

model.compile(optimizer=RMSprop(learning_rate=0.001),loss='categorical_crossentropy',metrics=['accuracy'])

accuracy=model.fit(X,y,epochs=100)

model.save('winemodel.h5')
accuracy=accuracy.history['accuracy'][-1:][0]
print(accuracy)

