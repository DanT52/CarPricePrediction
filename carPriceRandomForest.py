# %%
# Imports
import pandas as pd
import numpy as np
import re
import cloudpickle
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.ensemble import RandomForestRegressor

# %%
# Load Dataset
df_train = pd.read_csv('train.csv')
df_test = pd.read_csv('test.csv')

# %%
# check for missing values and get percent of missing values
missing_values = df_train.isnull().sum()
missing_percentage = (missing_values / len(df_train)) * 100

# display percents of columns with missing values
print(missing_percentage[missing_percentage > 0])

# total number of columns
print(f"Total number of rows in df_train: {len(df_train)}")

# %%
# print the unique values of columns with missing data
print(df_train['fuel_type'].value_counts(dropna=False))
print(df_train['accident'].value_counts(dropna=False))
print(df_train['clean_title'].value_counts(dropna=False))

# %%
# Filling in missing values
def fix_missing(df):
    df['fuel_type'] = df['fuel_type'].fillna('Electric')
    df['fuel_type'] = df['fuel_type'].replace(['–', 'not supported'], 'Electric')
    df['accident'] = df['accident'].fillna('None reported')
    df['accident'] = df['accident'].apply(lambda x: 1 if x != 'None reported' else 0)
    df['clean_title'] = df['clean_title'].fillna('missing')
    df['clean_title'] = df['clean_title'].apply(lambda x: 1 if x != 'missing' else 0)
    return df

df_train = fix_missing(df_train)
df_test = fix_missing(df_test)

# %%
# extracting data from engine column
def extract_car_specs(entry):
    hp_match = re.search(r'(\d+\.?\d*)HP', entry)
    l_match = re.search(r'(\d+\.?\d*)\s*(?:L|Liter)', entry, re.IGNORECASE)
    cyl_match = re.search(r'(?:(\d+)\s+Cylinder)|(?:V(\d+))', entry, re.IGNORECASE)
    hp = float(hp_match.group(1)) if hp_match else None
    L = float(l_match.group(1)) if l_match else None
    if cyl_match:
        cylCount = int(cyl_match.group(1) or cyl_match.group(2))
    else:
        cylCount = None
    electric = 1 if 'electric' in entry.lower() else 0
    turbo = 1 if 'turbo' in entry.lower() else 0
    return hp, L, cylCount, electric, turbo

def process_engine_column(df):
    df[['hp', 'L', 'cylCount', 'electric', 'turbo']] = df['engine'].apply(lambda x: pd.Series(extract_car_specs(x)))    
    return df

process_engine_column(df_train)
process_engine_column(df_test)

df_train['hp'] = df_train['hp'].fillna(df_train['hp'].mean())
df_train['L'] = df_train['L'].fillna(df_train['L'].mean())
df_train['cylCount'] = df_train['cylCount'].fillna(df_train['cylCount'].mean())
df_test['hp'] = df_test['hp'].fillna(df_train['hp'].mean())
df_test['L'] = df_test['L'].fillna(df_train['L'].mean())
df_test['cylCount'] = df_test['cylCount'].fillna(df_train['cylCount'].mean())

# %%
# extracting data from transmission column
def extract_car_transmission(entry):
    speed_match = re.search(r'(\d+)', entry)
    speed = int(speed_match.group(1)) if speed_match else None
    at = 0
    mt = 0
    manual_transmissions = ["M/T", "Manual", "Mt"]
    if any(manual in entry for manual in manual_transmissions):
        mt = 1
    else:
        at = 1
    return speed, mt, at

def process_transmission_column(df):
    df[['trans_speed', 'manual', 'automatic']] = df['transmission'].apply(lambda x: pd.Series(extract_car_transmission(x)))    
    return df

process_transmission_column(df_train)
process_transmission_column(df_test)

df_train['trans_speed'] = df_train['trans_speed'].fillna(df_train['trans_speed'].mode())
df_test['trans_speed'] = df_test['trans_speed'].fillna(df_train['trans_speed'].mode())

# %%
# change model year to model age
df_train['model_age'] = 2025 - df_train['model_year']
df_test['model_age'] = 2025 - df_test['model_year']
df_train.drop('model_year', axis=1, inplace=True)
df_test.drop('model_year', axis=1, inplace=True)

# %%
# handle color columns
df_test['ext_col'] = df_test['ext_col'].replace(['–'], 'Black')
df_test['int_col'] = df_test['int_col'].replace(['–'], 'Black')
df_train['ext_col'] = df_train['ext_col'].replace(['–'], 'Black')
df_train['int_col'] = df_train['int_col'].replace(['–'], 'Black')

# %%
# Target Encoding
categorical_features = ['brand', 'model', 'fuel_type', 'ext_col', 'int_col', 'engine', 'transmission']
most_common_values = {col: df_train[col].mode()[0] for col in df_train.columns}
category_means = {}

for cat in categorical_features:
    cat_means = df_train.groupby(cat)['price'].mean().to_dict()
    most_common_value = most_common_values[cat]
    cat_means['__most_common__'] = cat_means.get(most_common_value, df_train['price'].mean())
    category_means[cat] = cat_means

def target_encode_with_saved_means(df, cat_col, means_dict):
    most_common = means_dict.get('__most_common__', 0)
    df[cat_col] = df[cat_col].map(means_dict).fillna(most_common)

for cat in categorical_features:
    target_encode_with_saved_means(df_test, cat, category_means[cat])
    target_encode_with_saved_means(df_train, cat, category_means[cat])

with open('category_means.pkl', 'wb') as f:
    cloudpickle.dump(category_means, f)

with open('most_common_values.pkl', 'wb') as f:
    cloudpickle.dump(most_common_values, f)
print("Category means and most common values saved successfully!")

# %%
# column selector from previous projects
class SelectColumns(BaseEstimator, TransformerMixin):
    def __init__(self, columns):
        self.columns = columns
    def fit(self, xs, ys=None):
        return self
    def transform(self, xs):
        return xs[self.columns]

# %%
# Training the Random Forest Regressor Model
df_train = df_train[(df_train['price'] <= 500000)]
X = df_train.drop(['id', 'price'], axis=1)
y = df_train['price']
all_columns = [col for col in df_train.columns if col not in ['price', 'id']]

pipeline = Pipeline([
    ('select_columns', SelectColumns(['brand', 'model'])),
    ('scaler', StandardScaler()),
    ('regressor', RandomForestRegressor(random_state=42))
])

param_grid = {
    'select_columns__columns': [all_columns],
    'regressor__n_estimators': [250],
    'regressor__max_depth': [20],
    'regressor__min_samples_split': [7],
    'regressor__min_samples_leaf': [3]
}

grid_search = GridSearchCV(pipeline, param_grid, cv=5, scoring='neg_mean_squared_error', n_jobs=-1)
grid_search.fit(X, y)

best_model = grid_search.best_estimator_

# %%
# get the results!
best_rmse = np.sqrt(-grid_search.best_score_)
print(f'Root Mean Squared Error: {best_rmse}')
print(f'Best Parameters: {grid_search.best_params_}')

# save the model using cloudpickle
with open('best_model.pkl', 'wb') as f:
    cloudpickle.dump(best_model, f)
print("Best model saved successfully!")