# -*- coding: utf-8 -*-
"""
Created on Sun Jul 28 16:20:58 2024

@author: nandks
"""

import pandas as pd
import numpy as np
import holidays
from catboost import CatBoostRegressor, Pool
import pickle

# Define the path to the directory containing the files
base_path = r"C:\Users\nandks\Case study"

# Define the list of file names for hbf and sb
hbf_files = [
    f'{base_path}\\freiburgambahnhof-2016.csv',
    f'{base_path}\\freiburgambahnhof-2017.csv',
    f'{base_path}\\freiburgambahnhof-2018.csv',
    f'{base_path}\\freiburgambahnhof-2019.csv',
    f'{base_path}\\freiburgambahnhof-2020.csv',
    f'{base_path}\\freiburgambahnhof-2021.csv'
]
sb_files = [
    f'{base_path}\\freiburgschlossberg-2016.csv',
    f'{base_path}\\freiburgschlossberg-2017.csv',
    f'{base_path}\\freiburgschlossberg-2018.csv',
    f'{base_path}\\freiburgschlossberg-2019.csv',
    f'{base_path}\\freiburgschlossberg-2020.csv',
    f'{base_path}\\freiburgschlossberg-2021.csv'
]

# Function to load and concatenate files
def load_and_concatenate(files):
    df_list = [pd.read_csv(file, header=None, names=["Date", "Parking Spaces Available"]) for file in files]
    return pd.concat(df_list, ignore_index=True)

# Function to transform data
def transform(df):
    df['Date'] = pd.to_datetime(df['Date']).dt.floor('T')
    df['Day'] = df['Date'].dt.day.astype(str)
    df['Month'] = df['Date'].dt.month.astype(str)
    df['Hour'] = df['Date'].dt.hour.astype(str)
    df['Minute'] = df['Date'].dt.minute.astype(str)
    df['Hour-Minute'] = df['Hour'] + '.' + df['Minute']
    df['DOW'] = df['Date'].dt.dayofweek.astype(str)
    df['Year'] = df['Date'].dt.year
    df['Weekend Flag'] = df['Date'].dt.weekday >= 5
    df['Weekend Flag'] = df['Weekend Flag'].astype(str)
    
    # Public Holiday in Freiburg, Germany
    de_holidays = holidays.Germany(state='BW')
    df['Public Holiday'] = df['Date'].dt.date.apply(lambda x: x in de_holidays)
    df['Public Holiday'] = df['Public Holiday'].astype(str)
    df = df.set_index('Date')
    return df

# Load and preprocess data
hbf = load_and_concatenate(hbf_files)
sb = load_and_concatenate(sb_files)
hbf = hbf[hbf['Parking Spaces Available'] < 245]

hbf1 = transform(hbf)
sb1 = transform(sb)

train_df_hbf = hbf1[hbf1['Year'] < 2021]
test_df_hbf = hbf1[hbf1['Year'] == 2021]

train_df_sb = sb1[sb1['Year'] < 2021]
test_df_sb = sb1[sb1['Year'] == 2021]

# Update features list to include 'Year'
features = ['Day', 'Month', 'Hour', 'Hour-Minute', 'DOW', 'Weekend Flag', 'Public Holiday', 'Year']
target = 'Parking Spaces Available'
categorical_features = ['Day', 'Month', 'Hour', 'DOW', 'Weekend Flag', 'Public Holiday']

# Create Pools for hbf
train_pool_hbf = Pool(train_df_hbf[features], train_df_hbf[target], cat_features=categorical_features)
test_pool_hbf = Pool(test_df_hbf[features], test_df_hbf[target], cat_features=categorical_features)

# Create Pools for sb
train_pool_sb = Pool(train_df_sb[features], train_df_sb[target], cat_features=categorical_features)

# Initialize and train the model for hbf
model_hbf = CatBoostRegressor(verbose=0)
model_hbf.fit(train_pool_hbf)

# Initialize and train the model for sb
model_sb = CatBoostRegressor(verbose=0)
model_sb.fit(train_pool_sb)

# Calculate residuals and standard deviation for hbf
train_predictions_hbf = model_hbf.predict(train_pool_hbf)
residuals_hbf = train_df_hbf[target] - train_predictions_hbf
std_dev_hbf = np.std(residuals_hbf)

# Calculate residuals and standard deviation for sb
train_predictions_sb = model_sb.predict(train_pool_sb)
residuals_sb = train_df_sb[target] - train_predictions_sb
std_dev_sb = np.std(residuals_sb)

# Save models and data to pickle files
with open('models_and_data.pkl', 'wb') as f:
    pickle.dump({
        'model_hbf': model_hbf,
        'train_df_hbf': train_df_hbf,
        'test_df_hbf': test_df_hbf,
        'std_dev_hbf': std_dev_hbf,
        'model_sb': model_sb,
        'train_df_sb': train_df_sb,
        'test_df_sb': test_df_sb,
        'std_dev_sb': std_dev_sb,
        'features': features,
        'categorical_features': categorical_features,
        'target': target
    }, f)
