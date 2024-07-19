#!/usr/bin/env python
# coding: utf-8

# In[1]:


# Importing the dependencies
import math
import pandas_datareader as web
import yfinance as yf
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential
from keras.layers import Dense, LSTM
import matplotlib.pyplot as plt
plt.style.use('fivethirtyeight')


# In[2]:


# Getting the Stock Quote
df = yf.download('AAPL', start = '2012-01-01', end = '2024-04-15')
df


# In[3]:


# Getting the number of rows and columns in the dataset
df.shape


# In[4]:


# Visualizing the closing price history
plt.figure(figsize = (16, 8))
plt.title('Close Price History')
plt.plot(df['Close'])
plt.xlabel('Date', fontsize = 18)
plt.ylabel('Close Price USD ($)', fontsize = 18)
plt.show()


# In[5]:


# Creating a new Dataframe with only 'Close' column
data = df.filter(['Close'])

# Converting the dataframe to numpy array
dataset = data.values

# Getting the number of row to train the model
training_data_len = math.ceil(len(dataset) * 0.8)


# In[6]:


training_data_len


# In[7]:


# Scale the data
scaler = MinMaxScaler(feature_range = (0, 1))
scaled_data = scaler.fit_transform(dataset)


# In[8]:


scaled_data


# In[9]:


# Creating the training dataset and the scaled training dataset
train_data = scaled_data[0: training_data_len, :]


# In[10]:


# Spliting the data into X_train and Y_train dataset
X_train = []
Y_train = []
for i in range(70, len(train_data)):
    X_train.append(train_data[i - 70: i, 0])
    Y_train.append(train_data[i, 0])
    if i <= 70:
        print(X_train)
        print(Y_train)
        print()


# In[11]:


# Converting the X_train and Y_train to numpy arrays
X_train, Y_train = np.array(X_train), np.array(Y_train)


# In[12]:


# Reshape the data
X_train = np.reshape(X_train, (X_train.shape[0], X_train.shape[1], 1))
X_train.shape


# In[13]:


# Building the LSTM model
model = Sequential()
model.add(LSTM(60, return_sequences = True, input_shape = (X_train.shape[1], 1)))
model.add(LSTM(60, return_sequences = False))
model.add(Dense(30))
model.add(Dense(1))


# In[14]:


# Compiling the model
model.compile(optimizer = 'adam', loss = 'mean_squared_error')


# In[15]:


try:
    # Training the model
    model.fit(X_train, Y_train, batch_size=1, epochs=1)
except Exception as e:
    print("Error occurred during training:")
    print(e)


# In[16]:


# Creating the testing dataset and creating a new array containing scaled values
test_data = scaled_data[training_data_len - 70: , :]
X_test = []
Y_test = dataset[training_data_len: , :]
for i in range(70, len(test_data)):
    X_test.append(test_data[i - 70: i, 0])


# In[17]:


# Convert the data to a numpy array
X_test = np.array(X_test)


# In[18]:


# Reshape the data
X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], 1))


# In[19]:


# Getting the models predicted price value
predictions = model.predict(X_test)
predictions = scaler.inverse_transform(predictions)


# In[20]:


# Getting the root mean squared error (RMSE)
rmse = np.sqrt(np.mean(predictions - Y_test) ** 2)
rmse


# In[21]:


# Plotting the data
train = data[ :training_data_len]
valid = data[training_data_len: ]
valid['Predictions'] = predictions

# Visualizing the data
plt.figure(figsize = (16, 8))
plt.title('Model')
plt.xlabel('Date', fontsize = 18)
plt.ylabel('Close Price USD ($)', fontsize = 18)
plt.plot(train['Close'])
plt.plot(valid[['Close', 'Predictions']])
plt.legend(['Train', 'Val', 'Predictions'], loc = 'lower right')
plt.show()


# In[22]:


# Showing the valid and predicted prices
valid


# In[23]:


# Get the quote
apple_quote = yf.download('AAPL', start = '2012-01-01', end = '2024-04-15')


# In[24]:


# Creating new Dataframe
new_df = apple_quote.filter(['Close'])


# In[26]:


# Getting the last 60 day Closing price values and converting the dataframe to an array
last_60_days = new_df[-60: ].values


# In[28]:


# Scale the data to be valued between 0 and 1
last_60_days_scaled = scaler.transform(last_60_days)


# In[29]:


# Creating an empty list
x_test = []

# Appending the past 60 days
x_test.append(last_60_days_scaled)

# Converting the x_test dataset to numpy array
x_test = np.array(x_test)

# Reshaping the data
x_test = np.reshape(x_test, (x_test.shape[0], x_test.shape[1], 1))


# In[31]:


# Getting the predicted scaled price
pred_price = model.predict(x_test)
pred_price = scaler.inverse_transform(pred_price)
pred_price


# In[32]:


apple_quote2 = yf.download('AAPL', start = '2012-01-01', end = '2024-04-15')
apple_quote2['Close']

