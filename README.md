# California_housing_price_prediction
### **Summary of Regression Analysis for California House Price Prediction**

#### **1. Data Preprocessing**
- **Dataset**: California housing dataset from Kaggle.
- **Sanity Checks**:
  - No duplicate values.
  - Missing values in `total_bedrooms` were imputed using the median.
  - Identified outliers in `total_rooms`, `total_bedrooms`, `population`, and `households`.
- **Outlier Treatment**:
  - Applied the 95th percentile Winsorization to `total_rooms` and `total_bedrooms` to limit extreme values.
- **Feature Engineering**:
  - Created new derived features:
    - `rooms_per_household`
    - `bedrooms_per_house`
    - `population_per_household`
  - Performed **K-Means clustering** (5 clusters) based on `longitude` and `latitude`.
- **Feature Selection**:
  - High correlation found between `total_rooms`, `total_bedrooms`, `households`, and `population`, leading to multicollinearity.
  - Variance Inflation Factor (VIF) analysis helped reduce multicollinearity by dropping `total_rooms`, `total_bedrooms`, `households`, `longitude`, and `latitude`.

#### **2. Exploratory Data Analysis (EDA)**
- **Correlation Analysis**:
  - `median_income` showed the strongest positive correlation with `median_house_value`.
  - Scatter plots confirmed `median_income` as the most influential predictor.
- **Encoding**:
  - `ocean_proximity` was label-encoded.

#### **3. Randomforest Regression implementation **
- **Train-Test Split**:
  - Data split into **80% training** and **20% testing**.
- **Pipeline with Standardization & Random Forest Regression**:
  - Used `Pipeline` to apply **StandardScaler** and **RandomForestRegressor**.
  - Model Evaluation:
    - RMSE: **48,925.28**
    - MAE: **32,182.05**
    - R²: **0.8268**

#### **4. XGBoost Implementation (Pending)**
- Planned **GridSearchCV/RandomizedSearchCV** for **XGBoost hyperparameter tuning**.
- Model will be saved using **Pickle**.

### **Conclusion**
- **Random Forest Regressor performed well**, achieving an **R² of 0.8268**.
- **Next Steps**:
  - Optimize the model with **XGBoost and hyperparameter tuning**.
  - Deploy the model using **Flask or FastAPI**.
