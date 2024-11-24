# %%
# Imports
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, GridSearchCV, RandomizedSearchCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_squared_error, r2_score
import cloudpickle
from sklearn.ensemble import RandomForestRegressor

# %%
# Load Dataset
df_train = pd.read_csv('train.csv')
df_test = pd.read_csv('test.csv')
df_train.head()

# %% [markdown]
# # Data Exploring
# 
# - `engine` seems like it could have data extracted from it
# - `transmission` could probably also have data extracted from it

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

# %% [markdown]
# # Filling in missing values
# - fill in missing NaN, '-' and 'not supported' `fuel_type` values with  *Electric* 
#     - I tested this and the average value of the NaN, - and not supported values is higher than gasoline so I assume this means that those cars are electric
# - Turn `accident` column into a binary column with 1 for accident reported and 0 otherwise
# - Turn `clean_title` into a binary column also with 1 for Yes and 0 for NaN

# %%
def fix_missing(df):
    # assume that if entry was NaN or '-' or 'not supported' then the car was electric or something
    df['fuel_type'] = df['fuel_type'].fillna('Electric')
    df['fuel_type'] = df['fuel_type'].replace(['–', 'not supported'], 'Electric')

    # make ommisions of clean title or accident record assume there was none to report
    df['accident'] = df['accident'].fillna('None reported')
    df['accident'] = df['accident'].apply(
        lambda x: 1 if x != 'None reported' else 0
    )

    df['clean_title'] = df['clean_title'].fillna('missing')
    df['clean_title'] = df['clean_title'].apply(
        lambda x: 1 if x != 'missing' else 0
    )
    
    return df

# apply function to test and train df
df_train = fix_missing(df_train)
df_test = fix_missing(df_test)

# %%
# Taking a look at the spread of the prices
highest_price = df_train['price'].max()
lowest_price = df_train['price'].min()
average_price = df_train['price'].mean()

print(f"Highest Price: {highest_price}")
print(f"Lowest Price: {lowest_price}")
print(f"Average Price: {average_price}")

# filter out outliers
upper_bound = df_train['price'].quantile(0.999)
filtered_df = df_train[(df_train['price'] <= upper_bound)]

# Plot
plt.figure(figsize=(12, 8))
sns.histplot(filtered_df['price'], bins=50, kde=True, color='skyblue')
plt.title('Distribution of Price (Without Outliers)', fontsize=16)
plt.xlabel('Price', fontsize=14)
plt.ylabel('Frequency', fontsize=14)
plt.axvline(filtered_df['price'].max(), color='red', linestyle='dashed', linewidth=1, label=f'Highest Price: {filtered_df["price"].max()}')
plt.axvline(filtered_df['price'].min(), color='green', linestyle='dashed', linewidth=1, label=f'Lowest Price: {filtered_df["price"].min()}')
plt.axvline(filtered_df['price'].mean(), color='blue', linestyle='dashed', linewidth=1, label=f'Average Price: {filtered_df["price"].mean()}')
plt.legend()
plt.grid(True)
plt.show()

# %%

print(f"Number of rows in df_train: {df_train.shape[0]}")
print(f"Number of unique entries in 'engine' column: {df_train['engine'].nunique()}")
df_train.tail()

# %% [markdown]
# # extracting data from engine column
# - I see that the format of the engine column is inconsistent
# - Attempt to extract the following data: hp, L, cylCount, electric, turbo.
# - electric and turbo are binary columns, which are set if those keywords were found in the engine value.
# - If a value like hp is valued to be found it will be replaced my the mean hp found at the end.

# %%
import re

def extract_car_specs(entry):
    # match a number directly followed by HP "420.0HP"
    hp_match = re.search(r'(\d+\.?\d*)HP', entry)
    
    # matches "6.2L" or "2.0 Liter"
    l_match = re.search(r'(\d+\.?\d*)\s*(?:L|Liter)', entry, re.IGNORECASE)
    
    # matches "6 Cylinder" or "V6"
    cyl_match = re.search(r'(?:(\d+)\s+Cylinder)|(?:V(\d+))', entry, re.IGNORECASE)


    # Extract the values if matches are found
    hp = float(hp_match.group(1)) if hp_match else None
    L = float(l_match.group(1)) if l_match else None

    if cyl_match:
        # Determine which group matched for cylinder count
        cylCount = int(cyl_match.group(1) or cyl_match.group(2))
    else:
        cylCount = None

    electric = 0
    if 'electric' in entry.lower():
        electric = 1
    
    turbo = 0
    if 'turbo' in entry.lower():
        turbo = 1

    return hp, L, cylCount, electric, turbo


def process_engine_column(df):
    # apply extract function
    df[['hp', 'L', 'cylCount', 'electric', 'turbo']] = df['engine'].apply(lambda x: pd.Series(extract_car_specs(x)))    
    return df


process_engine_column(df_train)
process_engine_column(df_test)

# fill na columns with means.
df_train['hp'] = df_train['hp'].fillna(df_train['hp'].mean())
df_train['L'] = df_train['L'].fillna(df_train['L'].mean())
df_train['cylCount'] = df_train['cylCount'].fillna(df_train['cylCount'].mean())
df_test['hp'] = df_test['hp'].fillna(df_train['hp'].mean())
df_test['L'] = df_test['L'].fillna(df_train['L'].mean())
df_test['cylCount'] = df_test['cylCount'].fillna(df_train['cylCount'].mean())

df_train.head()



# %%
# get more details on transmission values
num_unique_trans = df_train['transmission'].nunique()
transmission_counts = df_train['transmission'].value_counts()
print(f"unique values in 'transmission' column: {num_unique_trans}")
print(transmission_counts)

# %% [markdown]
# # Extract data out of transmission
# - I saw we can extract the speeds and if its manual or automatic.
# - pretty much same process as the engine

# %%
# get stuff out of the transmission column
def extract_car_transmission(entry):
    # match any number in the transmission entry

    speed_match = re.search(r'(\d+)', entry)
    speed = int(speed_match.group(1)) if speed_match else None
    
    # Determine if the transmission is manual or automatic
    at = 0
    mt = 0
    manual_transmissions = ["M/T", "Manual", "Mt"]
    if any(manual in entry for manual in manual_transmissions):
        mt = 1
    else:
        at = 1
    
    return speed, mt, at

def process_transmission_column(df):
    # apply the extract function
    df[['trans_speed', 'manual', 'automatic']] = df['transmission'].apply(lambda x: pd.Series(extract_car_transmission(x)))    
    return df

process_transmission_column(df_train)
process_transmission_column(df_test)

# replace NaN values with most common value
df_train['trans_speed'] = df_train['trans_speed'].fillna(df_train['trans_speed'].mode())
df_test['trans_speed'] = df_test['trans_speed'].fillna(df_train['trans_speed'].mode())

df_train.head()

# %%
# change model year to model age

df_train['model_age'] = 2025 - df_train['model_year']
df_test['model_age'] = 2025 - df_test['model_year']

# drop origional columns model year
df_train.drop('model_year', axis=1, inplace=True)
df_test.drop('model_year', axis=1, inplace=True)

# %%
# learn what kind of values the colors can be
print("Unique External: ", df_train['ext_col'].nunique())
print("Unique Internal: ", df_train['int_col'].nunique())
print(df_train['ext_col'].value_counts())

# assume - values are just black
df_test['ext_col'] = df_test['ext_col'].replace(['–'], 'Black')
df_test['int_col'] = df_test['int_col'].replace(['–'], 'Black')
df_train['ext_col'] = df_train['ext_col'].replace(['–'], 'Black')
df_train['int_col'] = df_train['int_col'].replace(['–'], 'Black')

# %% [markdown]
# There can be very many unique colors, I will leave them as is, but a more advanced strategy could be to try to group all the unique color names into like a small list of similified names merging colors that are similar.

# %%
#Heatmap
numerical = ['milage', 'hp', 'L', 'cylCount', 'trans_speed', 'model_age']
columns_for_correlation = numerical + ['price']

# plot the heatmap
correlation_matrix = df_train[columns_for_correlation].corr()
plt.figure(figsize=(10, 8))
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt=".2f")
plt.title("Correlation Heatmap of Numerical Features with Price")
plt.show()

# %%
# bar graphs for all columns categorical columns
categorical_columns = ['brand', 'model', 'fuel_type', 'ext_col', 'int_col', 
                       'accident', 'clean_title', 'electric', 'manual', 'automatic', 'model_age', 'engine', 'transmission', 'trans_speed']

# get the top 8 values, calculate their mean prices
def top_categories_correlation(df, col, target='price', top_n=8):
    top_categories = df[col].value_counts().nlargest(top_n).index
    filtered_data = df[df[col].isin(top_categories)]
    mean_prices = filtered_data.groupby(col)[target].mean()
    return mean_prices

# bar graph for each categorical coulnm
for col in categorical_columns:
    plt.figure(figsize=(8, 6))
    correlation_data = top_categories_correlation(df_train, col, target='price')
    
    correlation_df = correlation_data.reset_index()
    correlation_df.columns = [col, 'Mean Price']
    sns.barplot(
        data=correlation_df,
        x=col,
        y='Mean Price',
        hue=col,  # assign the x variable to hue
        dodge=False,
        palette="coolwarm",
        legend=False
    )

    plt.title(f"Mean Price by {col.title()} Categories")
    plt.xlabel(col.title())
    plt.ylabel("Mean Price")
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    plt.show()

# %% [markdown]
# # Encoding categorical values
# - I am using a target encoding
#     - so each brand ect. is replaced with the average price for that value.
#     - i also save the most common values for each category, since in the test set there are some new categories if we see a new category it is replaced with the most common one.

# %%
# Target Encoding
categorical_features = ['brand', 'model', 'fuel_type', 'ext_col', 'int_col', 'engine', 'transmission']

most_common_values = {col: df_train[col].mode()[0] for col in df_train.columns}
category_means = {}

# go though each categorical column
for cat in categorical_features:
    # group by the category and calculate mean price
    cat_means = df_train.groupby(cat)['price'].mean().to_dict()
    
    # add the most common value (mode) for the column as the fallback
    most_common_value = most_common_values[cat]  # Get the most common value
    cat_means['__most_common__'] = cat_means.get(most_common_value, df_train['price'].mean())  
    
    # save
    category_means[cat] = cat_means

def target_encode_with_saved_means(df, cat_col, means_dict):
    # map the categories to their means, using the mode for unknowns
    most_common = means_dict.get('__most_common__', 0)
    df[cat_col] = df[cat_col].map(means_dict).fillna(most_common)

# run for each category on both test and train
for cat in categorical_features:
    target_encode_with_saved_means(df_test, cat, category_means[cat])
    target_encode_with_saved_means(df_train, cat, category_means[cat])


# save category means and common values
with open('category_means.pkl', 'wb') as f:
    cloudpickle.dump(category_means, f)

with open('most_common_values.pkl', 'wb') as f:
    cloudpickle.dump(most_common_values, f)
print("Category means and most common values saved successfully!")

df_train.head()

# %%
# column selector from previous projects
from sklearn.base import BaseEstimator, TransformerMixin
# hyper params
class SelectColumns( BaseEstimator, TransformerMixin ):
    # pass the function we want to apply to the column 'SalePrice’
    def __init__( self, columns ):
        self.columns = columns
        # don't need to do anything
    def fit( self, xs, ys, **params ):
        return self
    # actually perform the selection
    def transform( self, xs ):
        return xs[ self.columns ]

# %% [markdown]
# # Training the Random Forest Regressor Model
# - I chose random forest regressor because it should do decently with so many categoricals.
# - random forests should also be somewhat rescilent to noise.
# - I tried removing some columns i thought were the worst predictors but the best scores still came from useing all columns.

# %%
# Training the model

#remove outliers
df_train = df_train[(df_train['price'] <= 500000)]

X = df_train.drop(['id', 'price'], axis=1)
y = df_train['price']

# gets rid of price and id 
all_columns = [col for col in df_train.columns if col not in ['price', 'id']]

# column subests to try
subset_columns_1 = [col for col in all_columns if col not in ['L', 'trans_speed', 'cylCount']]
subset_columns_2 = [col for col in all_columns if col not in ['L', 'trans_speed']]
subset_columns_3 = [col for col in all_columns if col not in ['L']]
subset_columns_4 = [col for col in all_columns if col not in ['manual']]
subset_columns_5 = [col for col in all_columns if col not in ['turbo']]
subset_columns_6 = [col for col in all_columns if col not in ['engine', 'transmission']]


# pipeline!
pipeline = Pipeline([
    ('select_columns', SelectColumns(['brand', 'model'])),
    ('scaler', StandardScaler()),
    ('regressor', RandomForestRegressor(random_state=42))
])

# Define the parameter grid for GridSearchCV
# param_grid = {
#     'select_columns__columns': [  all_columns, subset_columns_1, subset_columns_2, subset_columns_3, subset_columns_4, subset_columns_5, subset_columns_6],
#     'regressor__n_estimators': [100, 200, 250],
#     'regressor__max_depth': [10, 20, 25],
#     'regressor__min_samples_split': [2, 5, 7],
#     'regressor__min_samples_leaf': [1, 2, 3]
# }

#Best params
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

# Print the RMSE
# Convert the best score (negative MSE) to RMSE
best_rmse = np.sqrt(-grid_search.best_score_)
print(f'Root Mean Squared Error: {best_rmse}')
# Print the best parameters
print(f'Best Parameters: {grid_search.best_params_}')

# save the model using cloudpickle
with open('best_model.pkl', 'wb') as f:
    cloudpickle.dump(best_model, f)
print("Best model saved successfully!")

# %%
# generate the submission output

X_test = df_test.drop('id', axis=1)


y_pred = best_model.predict(X_test)
submission = pd.DataFrame({
    'id': df_test['id'],
    'price': y_pred
})

# Save the submission DataFrame to a csv
submission.to_csv('submission.csv', index=False)

submission.head()

# %%



