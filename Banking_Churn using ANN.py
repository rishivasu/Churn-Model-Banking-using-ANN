
# coding: utf-8

# # DATA PREPROCESSING

# In[ ]:


#Importing Libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')


# In[ ]:


#Importing the datset and features from the dataset
data=pd.read_csv('Churn_Modelling.csv')
X=data.iloc[:,3:13].values
y=data.iloc[:, 13].values


# In[ ]:


#Dealing with Categorical Data...
from sklearn.preprocessing import LabelEncoder,OneHotEncoder
labelencoder_x1=LabelEncoder()
X[:,1] = labelencoder_x1.fit_transform(X[:,1])
labelencoder_x2=LabelEncoder()
X[:,2] =labelencoder_x2.fit_transform(X[:,2])
#onehotencoder for dummy variables if more than 2 cateogory are in column
onehotencoder=OneHotEncoder(categorical_features=[1])
X=onehotencoder.fit_transform(X).toarray()
#Remove one dummy Variable due to avoid the Dummy Variable trap
X=X[:,1:]


# In[ ]:


#Splitting teh dataet into train and test
from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.2,random_state=0)


# In[ ]:


# len(X_train)
len(y_train)


# In[ ]:


#Apply feature scaling to Convert all data into particular range
from sklearn.preprocessing import StandardScaler
sc=StandardScaler()
X_train=sc.fit_transform(X_train)
X_test=sc.transform(X_test)


# # ANN MODELING

# In[ ]:


import keras


# In[ ]:


from keras.models import Sequential #Use to intialize the ANN model
from keras.layers import Dense      #Use to Create layers in Artificial neural network


# In[ ]:


#Intialize the neural network
classifier=Sequential()


# In[ ]:


#Add first layer and first hidden layer
classifier.add(Dense(output_dim=6 , init ='uniform' , activation='relu',input_dim=11))


# In[ ]:


#Second hidden layer
classifier.add(Dense(output_dim=6 , init ='uniform' , activation='relu'))


# In[ ]:


#Final hidden layer
classifier.add(Dense(output_dim=1 , init ='uniform' , activation='sigmoid'))


# In[ ]:


#Compiling the ANN model
classifier.compile(optimizer='adam'  , loss='binary_crossentropy' , metrics=['accuracy'])


# In[ ]:


#Fitting the ANN
classifier.fit(X_train,y_train,batch_size=10,epochs=100)


# In[ ]:


#Preicting the values
y_pred=classifier.predict(X_test)
y_pred=(y_pred > 0.5)


# In[ ]:


from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test,y_pred)


# In[ ]:


cm


# In[ ]:


(1539+138)/2000

