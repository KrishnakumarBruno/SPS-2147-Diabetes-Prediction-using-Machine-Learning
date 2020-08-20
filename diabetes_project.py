# Importing necessary libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Imporing diabetes dataset
dataset = pd.read_csv('diabetes.csv')

# Checking if any null value is available
print(dataset.isnull().any())

# Obtaining independent & dependent variables
x = dataset.iloc[:,0:8].values
y = dataset.iloc[:,8].values
print(type(y))

# Performing Label Encoding
from sklearn.preprocessing import LabelEncoder
en = LabelEncoder()
y = en.fit_transform(y)

# Splitting the model into training and testing 
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test = train_test_split(x,y, test_size = 0.4,random_state=0)

# Performing feature Scaling
from sklearn.preprocessing import StandardScaler
sd = StandardScaler()
x_train = sd.fit_transform(x_train)
x_test = sd.transform(x_test)


