# Importing necessary libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
import sklearn.metrics as met

# Function to convert 2d list to 1d list
def twoD_List_to_oneD_List(s):
    # initialize an empty 1d list
    lis1 = [ ]
    for i in range(len(s)):
        val = s[i][0]
        lis1.append(val)

    # return string   
    return lis1

# Z-Score method
def Z_Score(x):
    # Z-Score Outlier detection and removal method
    # Initialising & defining constants
    THRESHOLD = 3  # Setting the threshold for value so obtained

    finalData = {}

    print(x.info(verbose=True))
    # print(x.values)
    # print(x.shape)

    for _i in range(len(x.columns.values)):

        # getting the first column of independent variable
        column_name = x.columns[_i]

        # converting nd.array into 2D list
        arr = x[[column_name]].values
        list1 = arr.tolist()

        # return from function twoD_List_to_oneD_List()
        data = twoD_List_to_oneD_List(list1)
        # print(data)

        # Obtaining mean & standard deviation
        mean = np.mean(data) 
        std = np.std(data) 

        outlier = []        # empty list for outlier
        outlier_check = []  # emptty list for rechecking of outliers

        # Detecting the outliers
        for i in data: 
            z = (i-mean)/std 
            if z > THRESHOLD: 
                outlier.append(i)

        # if outlier == []:
        #     print("No outlier")
        # else:
        #     print('outlier in dataset is', outlier) 

        # Eliminating the outliers
        for i in outlier:
            for n,i_ in enumerate(data):
                if i == i_:
                    data[n] = round(mean)
        
        # Peforming a recheck on the outliers
        for i in data: 
            z = (i-mean)/std 
            if z > THRESHOLD: 
                outlier_check.append(i)

        # if outlier != []:
        #     print('outlier in dataset after eliminating: ', outlier_check)

        finalData[x.columns[_i]] = data

    df = pd.DataFrame(data=finalData)
    # print(df.info(verbose=True))
    return df.values

# Imporing diabetes dataset
dataset = pd.read_csv('diabetes.csv')

# Checking if any null value is available
print(dataset.isnull().any())

# Obtaining independent & dependent variables
x = dataset.iloc[:,0:8]
y = dataset.iloc[:,8].values
# print(type(y))

# Performing Label Encoding
en = LabelEncoder()
y = en.fit_transform(y)

# Performing Z-Score to get rid of outliers
x = Z_Score(x)

# Splitting the model into training and testing 
x_train,x_test,y_train,y_test = train_test_split(x,y, test_size = 0.4,random_state=0)

# Training the model
pipe = Pipeline([
     ('rescale', StandardScaler()),
     ('classifier',KNeighborsClassifier(n_neighbors=5,metric='minkowski',p=2))
    ])
pipe.fit(x_train,y_train)

# Testing the model
y_predict = pipe.predict(x_test)
# print(y_predict)

# evaluation of the prediction
# 1st evaluation accuracy_score
print(accuracy_score(y_test,y_predict))
# 2nd confusion matrix
cm = confusion_matrix(y_test,y_predict)
print(cm)
# 3rd roc/auc curve
fpr,tpr,threshold = met.roc_curve(y_test,y_predict)
roc_auc = met.auc(fpr,tpr)
print(roc_auc)
plt.title("Receiver Operating Characteristics")
plt.plot(fpr,tpr,label = 'AUC-%0.2f'%roc_auc, color='blue')
plt.legend()
plt.show()