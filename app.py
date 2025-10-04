# import Requirements
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression,Ridge,Lasso
from sklearn.ensemble import RandomForestRegressor,GradientBoostingRegressor
from sklearn.metrics import mean_squared_error,r2_score
import streamlit as st
import pickle

# Data Reading
df = pd.read_csv(r'C:\Project_3\youtube_ad_revenue_dataset.csv')
df.drop_duplicates()

# Fill missing values
df['watch_time_minutes'] = df['watch_time_minutes'].fillna(df['watch_time_minutes'].mean())
df['comments'] = df['comments'].fillna(df['comments'].mean())
df['likes'] = df['likes'].fillna(df['likes'].mean())

# Encode categorical features
category_map = {'Education': 1, 'Music': 2, 'Tech': 3, 'Entertainment': 4, 'Gaming': 5, 'Lifestyle': 6}
device_map = {'TV': 1, 'Mobile': 2, 'Desktop': 3, 'Tablet': 4}
country_map = {'CA': 1, 'DE': 2, 'IN': 3, 'AU': 4, 'UK': 5, 'US': 6}   

# Mapping the categorical features to numerics
df['category'] = df['category'].map(category_map)
df['device'] = df['device'].map(device_map)
df['country'] = df['country'].map(country_map)

# Dropping of the unique features in Data_Frame
df = df.drop(columns=['video_id', 'date'])

#  Create new features (e.g., engagement rate = (likes + comments) / views).
df['engagement_rate'] = (df['likes'] + df['comments']) / df['views']

# Creating independent and Dependent Variables
X = df.drop(columns=['ad_revenue_usd'])
y = df['ad_revenue_usd']

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Define numeric and categorical columns
num_cols = ['views', 'likes', 'comments', 'watch_time_minutes',
                'video_length_minutes', 'subscribers', 'engagement_rate']
cat_cols = ['category', 'device', 'country']

# Preprocessing (Scaling the Numerical Features)
scaler = StandardScaler()
X_train[num_cols] = scaler.fit_transform(X_train[num_cols])
X_test[num_cols] = scaler.transform(X_test[num_cols])

#  models to be performed
models = {
        "Linear Regression": LinearRegression(),
        "Random Forest": RandomForestRegressor(n_estimators=200, random_state=42),
        "Gradient Boosting": GradientBoostingRegressor(n_estimators=200, random_state=42),
        "Ridge": Ridge(alpha=1.0),
        "Lasso": Lasso(alpha=1.0)
    }

# Train and evaluate each model

best_r2_score = -float('inf')
best_model = None
best_model_name = None

for name, model in models.items():
    model.fit(X_train, y_train)
    y_predicted = model.predict(X_test)
    mse = mean_squared_error(y_test, y_predicted)
    r2 = r2_score(y_test, y_predicted)
    rmse = mse ** 0.5
    print(f'{name} : mse = {mse:.4f} , r2 = {r2:.4f} , rmse = {rmse:.4f}')
    
    if r2 > best_r2_score:
        best_r2_score = r2         
        best_model = model
        best_model_name = name

print(f"\n✅ Best Model: {best_model_name} (R² = {best_r2_score:.4f})")

        
# Save best model & scaler
pickle.dump(best_model, open("best_model.pkl", "wb"))
pickle.dump(scaler, open("scaler.pkl", "wb"))