#!/usr/bin/env python
# coding: utf-8

# # Importing neccesary python modules

# In[1]:


import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime
from datetime import timedelta
import tensorflow as tf


# # Importing Dataset

# In[2]:


# To Import Data from google drive (authentication needed)
#from google.colab import drive 
#drive.mount('/content/gdrive')
#Data=pd.read_csv('gdrive/My Drive/Personal Data Science Projects/Apple Stocks Predictions/AAPL.csv', parse_dates=['Date'])
Data=pd.read_csv('AAPL.csv', parse_dates=['Date'])


# ###  Droppping Past Stocks data ( Data on year 2017 and less )

# In[3]:


Data.head()
Data = Data[ Data['Date']>=datetime(2016,1,1) ]


# ### Renaming the Data Column

# In[4]:


Data['date'] = Data[['Date']]
Data['close'] = Data['Close']
Data.drop(columns=['Date','Close'],inplace=True)
df = Data.copy()


# In[5]:


df


# # Visualization of the Data ( Close_Price vs Date )
# 

# In[6]:


fig = px.line( x = Data['date'],y=Data['close'] )
fig.show()


# *** 
# ***
# # Multivariate Time Series Predictions using LTSM 

# In[7]:


df_date = df[['date']]
df_close = df[['close']]
df.drop(columns=['date'],inplace=True)
df.head()


# # Scaling the all the features 
# ### Last scaler of this for loop is of column 'close' which we need later for inverse scaling

# In[8]:


from sklearn.preprocessing import StandardScaler
for feature in list(df.columns.values):
  scaler = StandardScaler()
  scaler.fit( df[[feature]] )
  df[feature] = scaler.transform( df[[feature]] )


# # Defining a function to create Timestaped Data Matrix in which window size can be given as input
# 
# ### This function includes all the feature available for creating matrix unlike in case of Univariate (where we use only one column i.e 'Close Price' )

# In[9]:


def Create_Timestaped_Data_Matrix_Multi( df, size ):
  X,y,dates = [],[],[]
  for i in range(df.shape[0]-size):
    X.append( np.asarray(df.values[i:i+size]).astype(np.float64) )
    y.append( df['close'].values[i+size] )
    #dates.append(df_date['date'][i+size])
    dates.append(df_date['date'][df_date['date'].index[i+size]])
  return np.array(X), np.array(y), dates

window_size = 50
X,y,dates = Create_Timestaped_Data_Matrix_Multi( df, size=window_size )


# In[10]:


X.shape, y.shape


# # Train-Test-Split

# In[11]:


test_size_ratio = 0.1
train_size = int((1-test_size_ratio)*len(y))
# Train Set
y_train = y[:train_size]
dates_train = dates[:train_size]
y_test = y[train_size:]
X_train = X[:train_size] 
# Test Set
dates_test = dates[train_size:]
X_test = X[train_size:]
y_train_original = Data['close'][window_size:train_size+window_size] # Storing original columns for plotting graph
y_test_original = Data['close'][train_size+window_size:] # Storing original columns for plotting graph
# Check length
# len(y_train)+len(y_test),len(y)


# In[12]:


## Splitting data according a particular date ( "split_date" )
# import datetime as dt
# split_date = df['date'][1100]
# split_date

# test = Data[ Data['date']>=split_date]
# train = Data[ Data['date']<Jan2019 ]
# test_date = Data[ Data['date']>=Jan2019].values
# train_date = Data[ Data['date']<Jan2019 ].values
# train.shape, test.shape

# X_train, y_train = Create_Timestaped_Data_Matrix( train, size=50 )
# X_test, y_test = Create_Timestaped_Data_Matrix( test, size=50 )


# ### Graph of Splitted Data

# In[13]:


fig = go.Figure()
fig.add_trace( go.Scatter( x=dates_train, y=y_train,mode='lines', name="Train Set" ) )
fig.add_trace( go.Scatter( x=dates_test, y=y_test,mode='lines', name="Test Set" ) )
fig.show()


# In[14]:


# # Reshaping and converting splited data into Dataframe for next step
# X_train = X_train.reshape(X_train.shape[0],X_train.shape[1])
# X_test = X_test.reshape(X_test.shape[0],X_test.shape[1])
# X_train = pd.DataFrame( X_train )
# X_test = pd.DataFrame( X_test )


# # Applying LSTM Model

# In[15]:


from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import LSTM
from tensorflow.keras.layers import Dropout
model = Sequential()
model.add( LSTM( 500, return_sequences =True, input_shape=(50,6) ) )
model.add( LSTM(100) ) 
model.add( Dense(1) )
model.compile( loss='mean_squared_error', optimizer='adam' )
model.summary()


# In[16]:


history = model.fit(X_train,y_train,validation_data = (X_test,y_test), epochs=25 , verbose=1)


# ### Saving the model

# In[17]:


model.save('/content/APPL_multivariate_lstm.h5')


# ###  Fucntion defined to change the shape of  Data ( from reshape(-1,1) to   ) 
# 

# In[18]:


def changing_shape( array ):
  temp = []
  for i in range(len(array)):
    temp.append(array[i][0])
  return temp


# ### Function defined for making Prediction using the model trained above, then apply inverse scaling and then changing the shape of output array

# In[19]:


def Prediction_Inverse_Scaling_Reshaping( model, X_train, X_test ):
  # Predicting values and Appyling Inverse Scaler
  y_pred = model.predict( X_test )
  y_pred = scaler.inverse_transform( y_pred )

  y_pred_train = model.predict( X_train )
  y_pred_train = scaler.inverse_transform( y_pred_train )

  # Reshaping
  y_pred = changing_shape(y_pred)
  y_pred_train = changing_shape( y_pred_train )

  return y_pred, y_pred_train


y_pred, y_pred_train = Prediction_Inverse_Scaling_Reshaping( model, X_train, X_test )


# In[20]:


fig = go.Figure()
fig.add_trace( go.Scatter( x=dates_train, y=y_train_original,mode='lines', name="Original Train Set" ) )
fig.add_trace( go.Scatter( x=dates_test, y=y_test_original,mode='lines', name="Original Test Set" ) )
fig.add_trace( go.Scatter( x=dates_train, y=y_pred_train, name="Predicted Train Set" ) )
fig.add_trace( go.Scatter( x=dates_test, y=y_pred,mode='lines', name="Predicted Test Set" ) )
fig.show()


# In[21]:


from sklearn.metrics import mean_squared_error
np.sqrt( mean_squared_error( y_pred, y_test_original ) )


# # Prediction for Next Days

# In[22]:


# Extracting last values values (of certain window size) from the df to use it for predicting stock price for next date
last = df.iloc[-window_size:].values 
pred_close = model.predict( last.reshape( 1, window_size, 6 ) )[0][0]
pred_close = scaler.inverse_transform( pred_close.reshape(-1,1) )[0][0]

# Adding one day to last date ( for adding "next" day date )
temp_date = dates_test[-1] + timedelta(days=1)

# Creating dataframe of 'next' and prediction on that day then appending them to df
future_pred_df =pd.DataFrame({'date':[temp_date],'close':[pred_close]})


# In[23]:


future_pred_df


# In[ ]:




