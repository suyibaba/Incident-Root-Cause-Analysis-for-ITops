#!/usr/bin/env python
# coding: utf-8

# #  Incident Root Cause Analysis 
# 
# Incident Reports in ITOps usually states the symptoms. Identifying the root cause of the symptom quickly is a key determinant to reducing resolution times and improving user satisfaction.

# Preprocessing Incident Data
# 
# ### Loading the Dataset

# In[1]:


import pandas as pd
import os
import tensorflow as tf

symptom_data = pd.read_csv("root_cause_analysis.csv")

print(symptom_data.dtypes)
symptom_data.head()


# ### Convert  data
# 
# Input data needs to be converted to formats that can be consumed by ML algorithms

# In[2]:


from sklearn import preprocessing
from sklearn.model_selection import train_test_split

label_encoder = preprocessing.LabelEncoder()
symptom_data['ROOT_CAUSE'] = label_encoder.fit_transform(
                                symptom_data['ROOT_CAUSE'])

np_symptom = symptom_data.to_numpy().astype(float)

X_data = np_symptom[:,1:8]

Y_data=np_symptom[:,8]
Y_data = tf.keras.utils.to_categorical(Y_data,3)

X_train,X_test,Y_train,Y_test = train_test_split( X_data, Y_data, test_size=0.10)

print("Shape of feature variables :", X_train.shape)
print("Shape of target variable :",Y_train.shape)


# ## Building and evaluating the model

# In[3]:


from tensorflow import keras
from tensorflow.keras import optimizers
from tensorflow.keras.regularizers import l2


EPOCHS=20
BATCH_SIZE=64
VERBOSE=1
OUTPUT_CLASSES=len(label_encoder.classes_)
N_HIDDEN=128
VALIDATION_SPLIT=0.2


model = tf.keras.models.Sequential()
#Add a Dense Layer
model.add(keras.layers.Dense(N_HIDDEN,
                             input_shape=(7,),
                              name='Dense-Layer-1',
                              activation='relu'))


model.add(keras.layers.Dense(N_HIDDEN,
                              name='Dense-Layer-2',
                              activation='relu'))


model.add(keras.layers.Dense(OUTPUT_CLASSES,
                             name='Final',
                             activation='softmax'))


model.compile(
              loss='categorical_crossentropy',
              metrics=['accuracy'])


model.summary()


model.fit(X_train,
          Y_train,
          batch_size=BATCH_SIZE,
          epochs=EPOCHS,
          verbose=VERBOSE,
          validation_split=VALIDATION_SPLIT)



print("\nEvaluation against Test Dataset :\n------------------------------------")
model.evaluate(X_test,Y_test)


# ##Predicting Root Causes

# In[4]:



import numpy as np

CPU_LOAD=1
MEMORY_LOAD=0
DELAY=0
ERROR_1000=0
ERROR_1001=1
ERROR_1002=1
ERROR_1003=0

prediction=np.argmax(model.predict(
    [[CPU_LOAD,MEMORY_LOAD,DELAY,
      ERROR_1000,ERROR_1001,ERROR_1002,ERROR_1003]]), axis=1 )

print(label_encoder.inverse_transform(prediction))


# In[1]:



print(label_encoder.inverse_transform(np.argmax(
        model.predict([[1,0,0,0,1,1,0],
                                [0,1,1,1,0,0,0],
                                [1,1,0,1,1,0,1],
                                [0,0,0,0,0,1,0],
                                [1,0,1,0,1,1,1]]), axis=1 )))


# In[ ]:




