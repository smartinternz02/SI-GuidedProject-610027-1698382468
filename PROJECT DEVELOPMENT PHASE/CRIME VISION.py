#!/usr/bin/env python
# coding: utf-8

# In[1]:


import cv2
import numpy as np
from keras.preprocessing import image_dataset_from_directory
from keras.models import Sequential
from keras.layers import Dense, Flatten, Dropout, GlobalAveragePooling2D
from keras.layers import Conv2D
from keras.optimizers.legacy import Adam
from keras.layers import MaxPooling2D
import os
from keras.applications import DenseNet121


# In[2]:


train_dir = 'C:/Users/VISHNU VARDHAN/Downloads/AIML PROJECT/Train'
test_dir = 'C:/Users/VISHNU VARDHAN/Downloads/AIML PROJECT/Test'


# In[3]:


crime_types = os.listdir(train_dir)
n = len(crime_types)
print("Number of crime categories: ", n)


# In[ ]:





# In[ ]:





# In[ ]:





# In[4]:


train_gen = image_dataset_from_directory(
    train_dir, image_size = (64,64), batch_size = 128, label_mode = "categorical",shuffle = True, seed = 12,
    validation_split=0.2, subset = 'training',
)


# In[5]:


val_set = image_dataset_from_directory(
    train_dir, image_size = (64,64), batch_size = 128, label_mode = "categorical",shuffle = True, seed = 12,
    validation_split=0.2, subset = 'validation',
)


# In[6]:


test_gen = image_dataset_from_directory(
    test_dir, image_size = (64,64), batch_size = 128,label_mode = "categorical",shuffle=False, seed=12, 
)


# In[7]:


def transfer_learning():
    base_model = DenseNet121(include_top=False, input_shape = (64,64,3), weights='imagenet')
    
    thr = 149
    for layers in base_model.layers[:thr]:
        layers.trainable=False
    for layers in base_model.layers[:thr]:
        layers.trainable=False
    return base_model


# In[8]:


def create_model():
    model = Sequential()
    
    base_model = transfer_learning()
    model.add(base_model)
    
    model.add(GlobalAveragePooling2D())
    
    model.add(Dense(256, activation='relu'))
    model.add(Dropout(0.2))
    
    model.add(Dense(512, activation='relu'))
    model.add(Dropout(0.2))
    
    model.add(Dense(1024, activation='relu'))
    
    model.add(Dense(n, activation='softmax'))
    
    model.summary
    
    return model


# In[9]:


model = create_model()
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics = ['accuracy'])


# In[10]:


model.summary()


# In[ ]:


model.fit(x=train_gen, validation_data=val_set, epochs = 1)


# In[ ]:


model.save('crime.h5')


# In[ ]:





# In[ ]:


from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import numpy as np
import cv2
import tensorflow as tf
from keras.preprocessing.image import img_to_array


# In[ ]:


model=tf.keras.models.load_model(r"C:/Users/VISHNU VARDHAN/crime.h5",compile=False)


# In[ ]:


img=image.load_img(r"C:/Users/VISHNU VARDHAN/Downloads/Fighting.png",target_size=(64,64))


# In[ ]:


img


# In[ ]:


x=image.img_to_array(img)
x


# In[ ]:


x=np.expand_dims(x,axis=0)


# In[ ]:


x.ndim


# In[ ]:


x.shape


# In[ ]:


pred=model.predict(x)


# In[ ]:


pred


# In[ ]:


{'Abuse':0,'Arrest':1, 'Arson':2, 'Assault':3, 'Burglary':4, 'Explosion':5, 'Fighting':6, 'NormalVideos':7,
'RoadAccidents':8, 'Robbery':9, 'Shooting':10, 'Shoplifting':11, 'Stealing':12, 'Vandalism':13}


# In[ ]:


pred_class=np.argmax(pred,axis=1)
pred_class[0]


# In[ ]:


index = ['Abuse','Arrest', 'Arson', 'Assault', 'Burglary', 'Explosion', 'Fighting', 'NormalVideos',
'RoadAccidents', 'Robbery', 'Shooting', 'Shoplifting', 'Stealing', 'Vandalism']
result=str(index[pred_class[0]])


# In[ ]:




