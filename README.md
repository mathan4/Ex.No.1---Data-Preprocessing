# Ex.No.1---Data-Preprocessing
## AIM:

To perform Data preprocessing in a data set downloaded from Kaggle

##REQUIPMENTS REQUIRED:
Hardware – PCs
Anaconda – Python 3.7 Installation / Google Colab /Jupiter Notebook

## RELATED THEORETICAL CONCEPT:

Kaggle :
Kaggle, a subsidiary of Google LLC, is an online community of data scientists and machine learning practitioners. Kaggle allows users to find and publish data sets, explore and build models in a web-based data-science environment, work with other data scientists and machine learning engineers, and enter competitions to solve data science challenges.

Data Preprocessing:

Pre-processing refers to the transformations applied to our data before feeding it to the algorithm. Data Preprocessing is a technique that is used to convert the raw data into a clean data set. In other words, whenever the data is gathered from different sources it is collected in raw format which is not feasible for the analysis.
Data Preprocessing is the process of making data suitable for use while training a machine learning model. The dataset initially provided for training might not be in a ready-to-use state, for e.g. it might not be formatted properly, or may contain missing or null values.Solving all these problems using various methods is called Data Preprocessing, using a properly processed dataset while training will not only make life easier for you but also increase the efficiency and accuracy of your model.

Need of Data Preprocessing :

For achieving better results from the applied model in Machine Learning projects the format of the data has to be in a proper manner. Some specified Machine Learning model needs information in a specified format, for example, Random Forest algorithm does not support null values, therefore to execute random forest algorithm null values have to be managed from the original raw data set.
Another aspect is that the data set should be formatted in such a way that more than one Machine Learning and Deep Learning algorithm are executed in one data set, and best out of them is chosen.


## ALGORITHM:
Importing the libraries
Importing the dataset
Taking care of missing data
Encoding categorical data
Normalizing the data
Splitting the data into test and train

## PROGRAM:
```
Done by : MATHAN S
Reg no : 212221040103
import pandas as pd
import numpy as np

df = pd.read_csv("Churn_Modelling.csv")
df

df.isnull().sum()

df.duplicated()

df.describe()

df['Exited'].describe()

""" Normalize the data - There are range of values in different columns of x are different. 

To get a correct ne plot the data of x between 0 and 1 

LabelEncoder can be used to normalize labels.
It can also be used to transform non-numerical labels to numerical labels.
"""
from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()

df1 = df.copy()

df1["Geography"] = le.fit_transform(df1["Geography"])
df1["Gender"] = le.fit_transform(df1["Gender"])

'''
MinMaxScaler - Transform features by scaling each feature to a given range. 
When we normalize the dataset it brings the value of all the features between 0 and 1 so that all the columns are in the same range, and thus there is no dominant feature.'''

from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler

scaler = MinMaxScaler()

df1[["CreditScore","Geography","Age","Tenure","Balance","NumOfProducts","EstimatedSalary"]] = pd.DataFrame(scaler.fit_transform(df1[["CreditScore","Geography","Age","Tenure","Balance","NumOfProducts","EstimatedSalary"]]))

df1

df1.describe()

# Since values like Row Number, Customer Id and surname  doesn't affect the output y(Exited).
# So those are not considered in the x values
X = df1[["CreditScore","Geography","Gender","Age","Tenure","Balance","NumOfProducts","HasCrCard","IsActiveMember","EstimatedSalary"]].values
print(X)

y = df1.iloc[:,-1].values
print(y)

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

print(X_train)
print("Size of X_train: ",len(X_train))

print(X_test)
print("Size of X_test: ",len(X_test))

X_train.shape
```

## OUTPUT:

### Dataset

![dataset](https://user-images.githubusercontent.com/109868924/195975945-80ea4012-8233-4375-9f75-1750758a9160.png)

## Checking for null values

![null](https://user-images.githubusercontent.com/109868924/195976068-beb58219-a29b-4808-89f1-2d2a4f587975.png)

## Checking for duplicate values

![duplicate](https://user-images.githubusercontent.com/109868924/195976094-35f35cbf-fe02-4c06-88b9-f60bee8bd643.png)

## Describing Data

![describe](https://user-images.githubusercontent.com/109868924/195976113-ffc5f01e-8378-41b7-8589-f729435e1a68.png)

## Checking for outliers in Exited Column

![outlier](https://user-images.githubusercontent.com/109868924/195976126-fb5c6af2-6ba4-4326-99ae-9e8ecab21b6b.png)

## Normalized Dataset

![normalized_data](https://user-images.githubusercontent.com/109868924/195976167-2f9327fe-a6db-4c19-b777-ca5c9a8f11e3.png)

## Describing Normalized Data

![Normalized_describe](https://user-images.githubusercontent.com/109868924/195976193-a1d3c0a8-237d-426d-ae53-81faf44fa1ad.png)

## X - Values

![x](https://user-images.githubusercontent.com/109868924/195976212-0e127805-91cd-4b82-9fea-22f94f0cba7f.png)

## Y - Value

![y](https://user-images.githubusercontent.com/109868924/195976226-4a7ea4e2-cffd-470c-b280-34bb62b2244f.png)

## X_train values

![x_train](https://user-images.githubusercontent.com/109868924/195976256-04bdc235-19c5-4ef9-a818-006b25a7f8ff.png)

## X_train Size

![x_train_size](https://user-images.githubusercontent.com/109868924/195976291-57f6a6f4-4b24-4ef0-9036-7208d85922f7.png)

## X_test values

![x_test](https://user-images.githubusercontent.com/109868924/195976323-e522f76a-5171-4b67-86aa-0cfda2a6a314.png)

## X_test Size

![x_test_size](https://user-images.githubusercontent.com/109868924/195976352-57e67e2a-548a-48b0-9ba7-c0c1d999dcb2.png)

## X_train shape

![x_train_shape](https://user-images.githubusercontent.com/109868924/195976375-ae170a53-16b3-4e98-9263-f0064e39e51a.png)


## RESULT
Data preprocessing is performed in a data set downloaded from Kaggle
