#!/usr/bin/env python
# coding: utf-8

# # TITANIC CLASSIFICATION

# In[1]:


# Importing dependencies
import numpy as np
import pandas as pd
import seaborn as sb
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score
from sklearn.linear_model import LogisticRegression


# In[3]:


# Loading the Dataset
df = pd.read_csv(r"E:\New folder (3)\Projects\Code_Alpha_Tititanic_Classification\train.csv")


# ## Data Exploration and Cleaning

# In[4]:


df.head()


# In[5]:


df.shape


# In[6]:


df.info()


# ## OBSERVATION
# 
# - DataFrame: Contains information on 891 Titanic passengers.
# - Data Types (dtypes):
#   - 5 integer columns (e.g., 'PassengerId', 'Survived').
#   - 2 float columns (e.g., 'Age', 'Fare').
#   - 5 object (string) columns (e.g., 'Name', 'Sex', 'Ticket').
# - Missing Values:
#   - Age: There are missing values in the 'Age' column. Specifically, 177 entries are missing.
#   - Cabin: Many missing values are present in the 'Cabin' column, 687 entries are missing.
#   - Embarked: There are a few missing values in the 'Embarked' column. Specifically, 2 entries are missing.
# - Memory Usage: Approximately 83.7 KB.

# In[7]:


df.isnull().sum()


# In[8]:


# Handling the missing values by dropping the irrelevant column i.e., CABIN
df = df.drop(columns = 'Cabin', axis = 1)


# In[9]:


# Handling the missing values in the Age with mean value
df['Age'].fillna(df['Age'].mean(), inplace = True)


# In[10]:


print(df['Embarked'].mode())


# In[11]:


# Handling the Missing Values in the Embarked with mode
df['Embarked'].fillna(df['Embarked'].mode()[0], inplace = True)


# In[12]:


df.isnull().sum()


# In[13]:


# Statistical Measures about the data
df.describe()


# ## Data Visualization

# In[14]:


sb.set()


# In[15]:


# Count Plot for 'Survived' column
sb.countplot(x = 'Survived', data = df)


# In[38]:


# Age distribution for survivors vs. non-survivors
sb.histplot(data = df, x = 'Age', hue = 'Survived', kde = True)
plt.title('Age Distribution for Survivors vs. Non-survivors')


# In[16]:


# Count Plot for 'Sex' column
sb.countplot(x = 'Sex', data = df)


# In[39]:


# Survival rate by embarkation port
sb.countplot(x = 'Embarked', hue = 'Survived', data = df)
plt.title('Survival Rate by Embarkation Port')


# In[17]:


# Number of surviver by gender
sb.countplot(x = 'Sex', hue = 'Survived', data = df)


# In[18]:


# Count Plot for Pclass column
sb.countplot(x = 'Pclass', data = df)


# In[20]:


# Number of surviver by Pclass
sb.countplot(x = 'Pclass', hue = 'Survived', data = df)


# ## Encoding Categorical Column

# In[21]:


df['Sex'].value_counts()


# In[22]:


df['Embarked'].value_counts()


# In[23]:


# Converting categorical columns
df.replace({'Sex': {'male': 0, 'female': 1}, 'Embarked': {'S': 0, 'C': 1, 'Q': 2}}, inplace = True)


# In[24]:


df.head()


# In[25]:


# Separating features and targets
X = df.drop(columns = ['PassengerId', 'Name', 'Ticket', 'Survived'], axis = 1)
Y = df['Survived']


# In[26]:


print(X)


# In[27]:


print(Y)


# ## Splitting the Data into Train and Test Data

# In[28]:


X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size = 0.2, random_state = 2)


# In[29]:


print(X.shape, X_train.shape, X_test.shape)


# ## Model Training
# 
# ### Logistic Regression

# In[30]:


model = LogisticRegression(solver = 'liblinear')


# In[31]:


# Training the Logistic Regression model with training data
model.fit(X_train, Y_train)


# ## Model Evaluation

# ### Accuracy Score

# In[32]:


# Accuracy on training data
X_train_prediction = model.predict(X_train)


# In[33]:


training_data_accuracy = accuracy_score(Y_train, X_train_prediction)
print('Accuracy Score of Training data:', training_data_accuracy)


# In[34]:


# Accuracy Score on test data
X_test_prediction = model.predict(X_test)


# In[35]:


testing_data_accuracy = accuracy_score(Y_test, X_test_prediction)
print('Accuracy Score of Testing data:', testing_data_accuracy)


# ### Model Accuracy: 78.21 %

# ### Precision

# In[36]:


precision = precision_score(Y_test, X_test_prediction)
print('Precision of the Logistic Regression model:', precision)


# ### Model Precision: 84.48 %

# ## Saving the Trained Model

# In[37]:


import pickle
filename = 'trained_model.sav'
pickle.dump(model, open(filename, 'wb'))


# In[41]:


# Loading the trained model
loaded_model = pickle.load(open(filename, 'rb'))

# Example data for prediction
example_data = pd.DataFrame({
    'Pclass': [3, 1],
    'Sex': [0, 1],  # 0 for male, 1 for female
    'Age': [25, 35],
    'SibSp': [1, 0],
    'Parch': [0, 2],
    'Fare': [7.5, 80], 
    'Embarked': [0, 1]  # S=0, C=1, Q=2
})

# Making predictions on example data
predictions = loaded_model.predict(example_data)

# Displaying the predictions
for i, prediction in enumerate(predictions):
    print(f"Passenger {i + 1}: {'Survived' if prediction == 1 else 'Not Survived'}")


# ## Factors most likely to lead to survival on the Titanic
# 
# 1. **Socio-Economic Status (Pclass)**:
#    - Passengers in the 1st class had a higher chance of survival compared to those in the 3rd class, indicating that socio-economic status played a significant role in survival.
# 
# 2. **Age**:
#    - Typically, younger individuals and children had higher survival rates compared to older passengers, as they were given priority during the evacuation.
# 
# 3. **Gender**:
#    - Female passengers had a much higher chance of survival compared to male passengers, which aligns with the "women and children first" policy followed during the Titanic disaster.

# ## OBSERVATION
# 1. **Dataset Information:**
#    - The dataset contains information on 891 Titanic passengers.
#    - Features include passenger ID, name, age, gender, ticket class, fare, port of embarkation, and whether they survived or not.
#    - There were missing values in the 'Age', 'Cabin', and 'Embarked' columns.
# 
# 2. **Data Cleaning:**
#    - Missing values in the 'Cabin' column were handled by dropping the column entirely.
#    - Missing values in the 'Age' column were filled with the mean age.
#    - Missing values in the 'Embarked' column were filled with the mode value.
# 
# 3. **Data Visualization:**
#    - Count plots were utilized to visualize the distribution of 'Survived', 'Sex', and 'Pclass' columns.
#    - Gender and passenger class were explored in relation to survival rates.
# 
# 4. **Model Training and Evaluation:**
#    - A Logistic Regression model was trained on the dataset.
#    - The accuracy of the model on both training and testing datasets was evaluated, achieving approximately 78.21% accuracy.
#    - The precision of the model is 84.48%.
# 
# 5. **Conclusion:**
#    - The analysis and modeling provide insights into the factors influencing survival on the Titanic.
#    - The trained model demonstrates a moderate level of predictive accuracy, but further improvements and evaluations could be conducted.
#    - The dataset and model can serve as a basis for further exploration and refinement in predictive modeling tasks.

# 
