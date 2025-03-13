#!/usr/bin/env python
# coding: utf-8

# In[1]:


get_ipython().system('pip install opendatasets')
get_ipython().system('pip install pandas')
import numpy as np
import seaborn as sns
import matplotlib.pyplot  as plt


# In[2]:


from pandas.plotting import scatter_matrix
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor

from sklearn import metrics


# # 1.) Loading the data

# In[3]:


import opendatasets as od
import pandas
od.download ( "https://www.kaggle.com/datasets/camnugent/california-housing-prices")


# In[4]:


import os
print("Current Working Directory:", os.getcwd())


# In[5]:


import pandas as pd
#Correct file path (ensure the extracted folder name matches this)
file = r'C:\Users\pooji\california-housing-prices/housing.csv'

df = pd.read_csv(file)  # Correct function for CSV files

# Display the content
df.head(5)


# In[6]:


df.tail(5)


# # 2,)Sanity check of data

# In[7]:


df.shape


# In[8]:


df.info()


# In[9]:


df.isna().sum() #finding the empty/null values


# In[10]:


#finding the percent of missing value (if >50 need to delete the column)
df.isna().sum() /df.shape[0]*100


# In[11]:


df.duplicated().sum()


# In[12]:


from sklearn.impute import SimpleImputer

# Median Imputation for 'total_bedrooms'
imputer = SimpleImputer(strategy='median')
df['total_bedrooms'] = imputer.fit_transform(df[['total_bedrooms']])


# In[13]:


df.isna().sum()


# In[14]:


#finding the unique values(garbage value in object data types)
for i in df.select_dtypes(include="object").columns:
    print(df[i].value_counts())


# # 3.) Exploratory Data Analysis

# In[15]:


df.describe()


# In[16]:


df.describe(include="object")


# In[17]:


sns.countplot(x="ocean_proximity",data=df)


# In[18]:


# Distribution plots for key numerical features
import warnings
warnings.filterwarnings("ignore")
for i in df.select_dtypes(include="number").columns:
    sns.histplot(data=df,x=i)
    plt.show()


# In[19]:


#histogram to understand numerical data distribution
df.hist(bins=10,figsize=(10,10))
plt.show()


# #Skewed Data (Right Skewed / Positive Skew)
# *total_rooms – Strong right skew
# *total_bedrooms – Strong right skew
# *population – Strong right skew
# *households – Strong right skew
# *median_income – Mild right skew
# #Less or Not Skewed Data
# *longitude – Bimodal distribution, not heavily skewed
# *latitude – Bimodal distribution, not heavily skewed
# *housing_median_age – Fairly uniform distribution
# *median_house_value – Slight right skew but not extreme

# In[20]:


#box plot to identify the outliers
plt.figure(figsize=(10, 6))
sns.boxplot(data=df[['median_income', 'total_rooms', 'total_bedrooms', 'population','households']])
plt.title('Boxplot for Outlier Detection')
plt.show()


# In[21]:


import numpy as np
df['total_rooms'] = np.log1p(df['total_rooms'])  # log1p handles zero values


# In[22]:


sns.distplot(df['total_rooms'])


# In[23]:


df['population'] = np.log1p(df['population'])  # log1p handles zero values


# In[24]:


sns.distplot(df['population'])


# In[25]:


Q1 = df['total_bedrooms'].quantile(0.25)
Q3 = df['total_bedrooms'].quantile(0.75)
IQR = Q3 - Q1
lower_bound = Q1 - 1.5 * IQR
upper_bound = Q3 + 1.5 * IQR
df = df[(df['total_bedrooms'] >= lower_bound) & (df['total_bedrooms'] <= upper_bound)]
df['total_bedrooms'] = np.where(df['total_bedrooms'] > upper_bound, upper_bound,
                                np.where(df['total_bedrooms'] < lower_bound, lower_bound, 
                                         df['total_bedrooms']))



# In[26]:


#sns.distplot(df['total_bedrooms'])


# In[27]:


sns.boxplot(data=df['total_bedrooms'])
plt.title('Total Bedrooms after outlier treatment')
plt.show()


# In[28]:


Q1 = df['households'].quantile(0.25)
Q3 = df['households'].quantile(0.75)
IQR = Q3 - Q1

# Filter out outliers
df = df[(df['households'] >= Q1 - 1.5 * IQR) & (df['households'] <= Q3 + 1.5 * IQR)]


# In[29]:


sns.boxplot(data=df['households'])
plt.title('Households after outlier treatment')
plt.show()


# In[30]:


#scatter plot to understand relationship
for i in ['longitude', 'latitude', 'housing_median_age', 'total_rooms',
       'total_bedrooms', 'population', 'households', 'median_income']:
    sns.scatterplot(data=df,x=i,y="median_house_value")
    plt.show()


# In[31]:


df.select_dtypes(include="number").columns


# In[32]:


df.select_dtypes(include="number").corr()


# In[33]:


plt.figure(figsize=(10, 8))
correlation_matrix = df.select_dtypes(include="number").corr()

sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', linewidths=0.5)
plt.title('Correlation Heatmap')
plt.show()


# # 4.)Feature Selection

# In[34]:


from statsmodels.stats.outliers_influence import variance_inflation_factor
import pandas as pd

# Assuming your feature data is in a DataFrame called 'X'
X = df[['total_rooms', 'total_bedrooms', 'population', 'households', 'median_income']]

# Add a constant for the intercept term
X['constant'] = 1  

# Calculate VIF for each feature
vif = pd.DataFrame({
    "Feature": X.columns,
    "VIF": [variance_inflation_factor(X.values, i) for i in range(X.shape[1])]
})

print(vif)


# In[35]:


# Creating derived features
df['rooms_per_household'] = df['total_rooms'] / df['households']
df['bedrooms_per_room'] = df['total_bedrooms'] / df['total_rooms']
df['population_per_household'] = df['population'] / df['households']


# In[36]:


# Dropping one of the original features
df.drop(['total_bedrooms'], axis=1, inplace=True)


# In[37]:


# Recalculate VIF
X = df[['rooms_per_household', 'bedrooms_per_room', 'population_per_household', 'median_income']]
X['constant'] = 1
vif = pd.DataFrame({
    "Feature": X.columns,
    "VIF": [variance_inflation_factor(X.values, i) for i in range(X.shape[1])]
})

print(vif)


# df['log_rooms_per_household'] = np.log1p(df['rooms_per_household'])
# 

# sns.distplot(df['log_rooms_per_household'])

# In[39]:


Q1 = df['rooms_per_household'].quantile(0.25)
Q3 = df['rooms_per_household'].quantile(0.75)
IQR = Q3 - Q1
df['rooms_per_household'] = np.where(
    df['rooms_per_household'] > Q3 + 1.5 * IQR,
    Q3 + 1.5 * IQR,
    df['rooms_per_household']
)


# In[40]:


sns.boxplot(data=df['rooms_per_household'])
plt.title('Room per household after outlier treatment')
plt.show()


# In[41]:


# Recalculate VIF
X = df[['rooms_per_household', 'bedrooms_per_room', 'population_per_household', 'median_income']]
X['constant'] = 1
vif = pd.DataFrame({
    "Feature": X.columns,
    "VIF": [variance_inflation_factor(X.values, i) for i in range(X.shape[1])]
})

print(vif)


# # Encoding categorical values

# In[42]:


labelEncoder = LabelEncoder()
print(df["ocean_proximity"].value_counts())
df["ocean_proximity"] = labelEncoder.fit_transform(df["ocean_proximity"])
df["ocean_proximity"].value_counts()
df.describe()


# In[45]:


df.head()


# In[71]:


from sklearn.model_selection import train_test_split

X = df.drop('median_house_value', axis=1)  # Features
y = df['median_house_value']               # Target

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)


# In[72]:


#check for rand_state
#X_train,X_test,y_train,y_test = train_test_split(df_ind,df_dep,test_size=0.2,random_state=42)
#print(X_train.head())
#print(X_test.head())
#print(y_train.head())
#print(y_test.head())
print("X_train shape {} and size {}".format(X_train.shape,X_train.size))
print("X_test shape {} and size {}".format(X_test.shape,X_test.size))
print("y_train shape {} and size {}".format(y_train.shape,y_train.size))
print("y_test shape {} and size {}".format(y_test.shape,y_test.size))


# In[73]:


X_train.head()


# StandardScalar

# In[74]:


#Standardize training and test datasets.
#==============================================================================
# Feature scaling is to bring all the independent variables in a dataset into
# same scale, to avoid any variable dominating  the model. Here we will not 
# transform the dependent variables.
#==============================================================================
independent_scaler = StandardScaler()
X_train = independent_scaler.fit_transform(X_train)
X_test = independent_scaler.transform(X_test)
print(X_train[0:5,:])
print("test data")
print(X_test[0:5,:])


# In[75]:


#initantiate the linear regression
linearRegModel = LinearRegression(n_jobs=-1)
#fit the model to the training data (learn the coefficients)
linearRegModel.fit(X_train,y_train)
#print the intercept and coefficients 
print("Intercept is "+str(linearRegModel.intercept_))
print("coefficients  is "+str(linearRegModel.coef_))


# In[76]:


#predict on the test data
y_pred = linearRegModel.predict(X_test)


# In[77]:


print(len(y_pred))
print(len(y_test))
print(y_pred[0:5])
print(y_test[0:5])


# In[78]:


test = pd.DataFrame({'Predicted':y_pred,'Actual':y_test})
fig= plt.figure(figsize=(16,8))
test = test.reset_index()
test = test.drop(['index'],axis=1)
plt.plot(test[:50])
plt.legend(['Actual','Predicted'])
sns.jointplot(x='Actual',y='Predicted',data=test,kind='reg',);


# In[79]:


print(np.sqrt(metrics.mean_squared_error(y_test,y_pred)))
print(np.sqrt(metrics.mean_squared_error(y_train,linearRegModel.predict(X_train))))
71147.87146118375


# In[80]:


dtReg = DecisionTreeRegressor(max_depth=9)
dtReg.fit(X_train,y_train)


# In[81]:


dtReg_y_pred = dtReg.predict(X_test)
dtReg_y_pred


# In[82]:


print(len(dtReg_y_pred))
print(len(y_test))
print(dtReg_y_pred[0:5])
print(y_test[0:5])


# In[83]:


print(np.sqrt(metrics.mean_squared_error(y_test,dtReg_y_pred)))


# In[84]:


test = pd.DataFrame({'Predicted':dtReg_y_pred,'Actual':y_test})
fig= plt.figure(figsize=(16,8))
test = test.reset_index()
test = test.drop(['index'],axis=1)
plt.plot(test[:50])
plt.legend(['Actual','Predicted'])
sns.jointplot(x='Actual',y='Predicted',data=test,kind="reg")


# In[85]:


rfReg = RandomForestRegressor(30)
rfReg.fit(X_train,y_train)


# In[86]:


rfReg_y_pred = rfReg.predict(X_test)
print(len(rfReg_y_pred))
print(len(y_test))
print(rfReg_y_pred[0:5])
print(y_test[0:5])


# In[87]:


print(np.sqrt(metrics.mean_squared_error(y_test,rfReg_y_pred)))


# In[88]:


test = pd.DataFrame({'Predicted':dtReg_y_pred,'Actual':y_test})
fig= plt.figure(figsize=(16,8))
test = test.reset_index()
test = test.drop(['index'],axis=1)
plt.plot(test[:50])
plt.legend(['Actual','Predicted'])
sns.jointplot(x='Actual',y='Predicted',data=test,kind="reg")


# SIMPLE LINEAR REGRESSION

# In[91]:


#Extract median_income 
X1 = df[['median_income']]   # Independent Variable
y1 = df['median_house_value'] # Dependent Variable


# In[92]:


#check for rand_state
X_train2,X_test2,y_train2,y_test2 = train_test_split(X1,y1,test_size=0.2,random_state=42)
#print(X_train.head())
#print(X_test.head())
#print(y_train.head())
#print(y_test.head())
print("X_train2 shape {} and size {}".format(X_train2.shape,X_train2.size))
print("X_test2 shape {} and size {}".format(X_test2.shape,X_test2.size))
print("y_train2 shape {} and size {}".format(y_train2.shape,y_train2.size))
print("y_test2 shape {} and size {}".format(y_test2.shape,y_test2.size))


# In[93]:


linReg2 = LinearRegression()
linReg2.fit(X_train2,y_train2)


# In[94]:


y_pred2 = linReg2.predict(X_test2)
print(len(y_pred2))
print(len(y_test2))
print(y_pred2[0:5])
print(y_test2[0:5])


# In[95]:


fig = plt.figure(figsize=(25,8))
plt.scatter(y_test2,y_pred2,marker="o",edgecolors ="r",s=60)
plt.scatter(y_train2,linReg2.predict(X_train2),marker="+",s=50,alpha=0.5)
plt.xlabel(" Actual median_house_value")
plt.ylabel(" Predicted median_house_value")


# In[ ]:




