#!/usr/bin/env python
# coding: utf-8

# In[1]:


#importing necessary libraries
get_ipython().system('pip install opendatasets')
get_ipython().system('pip install pandas')
import opendatasets as od
import os
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot  as plt


# In[2]:


#importing machine learning libraries
from pandas.plotting import scatter_matrix
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import numpy as np
from sklearn.impute import SimpleImputer
from sklearn import metrics


# # 1.) Loading the data

# In[3]:


od.download ( "https://www.kaggle.com/datasets/camnugent/california-housing-prices")


# In[4]:


print("Current Working Directory:", os.getcwd())


# In[5]:


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


# In[10]:


df.isna().sum() #finding the empty/null values


# In[11]:


#finding the percent of missing value (if >50 need to delete the column)
df.isna().sum() /df.shape[0]*100


# In[12]:


df.duplicated().sum()


# In[13]:


from sklearn.impute import SimpleImputer
# Median Imputation for 'total_bedrooms'
imputer = SimpleImputer(strategy='median')
df['total_bedrooms'] = imputer.fit_transform(df[['total_bedrooms']])


# In[14]:


df.isna().sum()


# In[15]:


#finding the unique values(garbage value in object data types)
for i in df.select_dtypes(include="object").columns:
    print(df[i].value_counts())


# # 3.) Exploratory Data Analysis

# In[16]:


df.describe()


# while looking at the statistics of the data above we can see  huge difference between 75% and max value in the columns of total_rooms, total_bedrooms,population,households,median_incme. SO, we have to do outlier treatment for these data sets

# In[17]:


df.describe(include="object")


# In[18]:


sns.countplot(x="ocean_proximity",data=df)


# Majority of the houses are situated at less than one hour didtances from the ocean, followed by inlan houses.

# In[21]:


df.plot(kind="scatter", x="longitude",y="latitude", c="median_house_value", cmap="jet", colorbar=True, legend=True, sharex=False, figsize=(10,7), s=df['population']/100, label="population", alpha=0.7)
plt.show()


# The above coordinates drew an outline of the densities of the houses situated in california, we can see that the bay area is densly populated and the inner areas are sparse and 

# In[22]:


# Distribution plots for key numerical features
import warnings
warnings.filterwarnings("ignore")
for i in df.select_dtypes(include="number").columns:
    sns.histplot(data=df,x=i)
    plt.show()


# In[20]:


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

# In[21]:


#box plot to identify the outliers
plt.figure(figsize=(10, 6))
sns.boxplot(data=df[['median_income', 'total_rooms', 'total_bedrooms', 'population','households']])
plt.title('Boxplot for Outlier Detection')
plt.show()


# we can see the extreme outliers in total_rooms,population and somewhat high outlies in total_bedrooms. They are all posivetly skewed.

# In[23]:


import numpy as np
df['total_rooms'] = np.log1p(df['total_rooms'])  # log1p handles zero values


# In[23]:


sns.distplot(df['total_rooms'])


# In[24]:


df['population'] = np.log1p(df['population'])  # log1p handles zero values


# In[25]:


sns.distplot(df['population'])


# In[26]:


Q1 = df['total_bedrooms'].quantile(0.25)
Q3 = df['total_bedrooms'].quantile(0.75)
IQR = Q3 - Q1
lower_bound = Q1 - 1.5 * IQR
upper_bound = Q3 + 1.5 * IQR
df = df[(df['total_bedrooms'] >= lower_bound) & (df['total_bedrooms'] <= upper_bound)]
df['total_bedrooms'] = np.where(df['total_bedrooms'] > upper_bound, upper_bound,
                                np.where(df['total_bedrooms'] < lower_bound, lower_bound, 
                                         df['total_bedrooms']))



# In[27]:


#sns.distplot(df['total_bedrooms'])


# In[28]:


sns.boxplot(data=df['total_bedrooms'])
plt.title('Total Bedrooms after outlier treatment')
plt.show()


# In[29]:


Q1 = df['households'].quantile(0.25)
Q3 = df['households'].quantile(0.75)
IQR = Q3 - Q1

# Filter out outliers
df = df[(df['households'] >= Q1 - 1.5 * IQR) & (df['households'] <= Q3 + 1.5 * IQR)]


# In[30]:


sns.boxplot(data=df['households'])
plt.title('Households after outlier treatment')
plt.show()


# In[31]:


#scatter plot to understand relationship
for i in ['longitude', 'latitude', 'housing_median_age', 'total_rooms',
       'total_bedrooms', 'population', 'households', 'median_income']:
    sns.scatterplot(data=df,x=i,y="median_house_value")
    plt.show()


# In[32]:


df.select_dtypes(include="number").columns


# In[33]:


df.select_dtypes(include="number").corr()


# In[34]:


plt.figure(figsize=(10, 8))
correlation_matrix = df.select_dtypes(include="number").corr()

sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', linewidths=0.5)
plt.title('Correlation Heatmap')
plt.show()


# # 4.)Feature Selection

# In[35]:


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


# In[36]:


# Creating derived features
df['rooms_per_household'] = df['total_rooms'] / df['households']
df['bedrooms_per_room'] = df['total_bedrooms'] / df['total_rooms']
df['population_per_household'] = df['population'] / df['households']
from sklearn.cluster import KMeans
kmeans = KMeans(n_clusters=5, random_state=42)
df['location_cluster'] = kmeans.fit_predict(df[['longitude', 'latitude']])


# In[37]:


# Dropping one of the original features
df.drop(['total_bedrooms'], axis=1, inplace=True)


# In[38]:


# Recalculate VIF
X = df[['rooms_per_household', 'bedrooms_per_room', 'population_per_household','location_cluster', 'median_income']]
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
X = df[['rooms_per_household', 'bedrooms_per_room', 'population_per_household','location_cluster', 'median_income']]

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


# In[43]:


df.head()


# In[55]:


df.dtypes


# In[ ]:


df.shape()


# In[ ]:


df.dtypes()


# In[44]:


from sklearn.model_selection import train_test_split

X = df.drop('median_house_value', axis=1)  # Features
y = df['median_house_value']               # Target

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)


# In[45]:


# Linear Regression Pipeline
lr_pipeline = Pipeline([
    ('scaler', StandardScaler()),
    ('regressor', LinearRegression())
])

# Decision Tree Pipeline
dt_pipeline = Pipeline([
    ('scaler', StandardScaler()),
    ('regressor', DecisionTreeRegressor(random_state=42))
])

# Random Forest Pipeline
rf_pipeline = Pipeline([
    ('scaler', StandardScaler()),
    ('regressor', RandomForestRegressor(random_state=42))
])
# Fit the Pipelines
lr_pipeline.fit(X_train, y_train)
dt_pipeline.fit(X_train, y_train)
rf_pipeline.fit(X_train, y_train)


# In[46]:


def evaluate_model(model, X_test, y_test):
    predictions = model.predict(X_test)
    rmse = np.sqrt(mean_squared_error(y_test, predictions))
    mae = mean_absolute_error(y_test, predictions)
    r2 = r2_score(y_test, predictions)

    return {'RMSE': rmse, 'MAE': mae, 'R²': r2}

# Evaluate models
results = {
    'Linear Regression': evaluate_model(lr_pipeline, X_test, y_test),
    'Decision Tree': evaluate_model(dt_pipeline, X_test, y_test),
    'Random Forest': evaluate_model(rf_pipeline, X_test, y_test)
}

# Display results
import pandas as pd
results_df = pd.DataFrame(results).T
print(results_df)


# In[47]:


get_ipython().system('pip install XGBoost')


# In[48]:


from xgboost import XGBRegressor
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import numpy as np
import pickle  # For saving the model


# In[49]:


xgb_model = XGBRegressor(random_state=42)

# Parameter Grid for Optimization
param_grid = {
    'n_estimators': [100, 200, 300],
    'learning_rate': [0.01, 0.1, 0.2],
    'max_depth': [3, 5, 7],
    'subsample': [0.8, 1],
    'colsample_bytree': [0.8, 1]
}


# In[50]:


grid_search = GridSearchCV(
    estimator=xgb_model,
    param_grid=param_grid,
    cv=5,                    # 5-fold cross-validation
    scoring='neg_root_mean_squared_error',
    n_jobs=-1,
    verbose=2
)

# Fit the grid search model
grid_search.fit(X_train, y_train)

# Best model from GridSearchCV
best_xgb_model = grid_search.best_estimator_
print("Best Parameters:", grid_search.best_params_)


# In[51]:


from xgboost import XGBRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import numpy as np

# Train XGBoost with best parameters
best_xgb_model = XGBRegressor(
    colsample_bytree=0.8,
    learning_rate=0.1,
    max_depth=7,
    n_estimators=300,
    subsample=1,
    random_state=42
)

best_xgb_model.fit(X_train, y_train)


# In[52]:


def evaluate_model(model, X_test, y_test):
    predictions = model.predict(X_test)
    rmse = np.sqrt(mean_squared_error(y_test, predictions))
    mae = mean_absolute_error(y_test, predictions)
    r2 = r2_score(y_test, predictions)
    return {'RMSE': rmse, 'MAE': mae, 'R²': r2}

# Evaluate XGBoost
xgb_results = evaluate_model(best_xgb_model, X_test, y_test)
print("XGBoost Results:", xgb_results)


# In[53]:


import pickle

# Save the trained XGBoost model
with open('best_xgb_model.pkl', 'wb') as model_file:
    pickle.dump(best_xgb_model, model_file)

print("Model saved successfully!")


# In[ ]:




