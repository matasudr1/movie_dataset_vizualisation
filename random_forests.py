# Import libraries
import pandas as pd
import numpy as np
import time
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import RandomizedSearchCV
import matplotlib.pyplot as plt
import seaborn as sns

# --- Measure total execution time ---
start_time = time.time()

# --- Load and Explore the Dataset ---
data = pd.read_csv(r"C:\Users\matas\OneDrive\Desktop\movie_project\Top Movies (Cleaned Data).csv")

# Display the first few rows to understand the data
print(data.head(10))
print(data.info())

# --- Data Preparation ---
# Clean column names
data.columns = data.columns.str.replace(' ', '_').str.replace(r'\W+', '', regex=True)

# --- Remove features that cause data leakage ---
# Drop the features directly related to the target variable
data = data.drop(['Worldwide_Box_Office_USD', 'International_Box_Office_USD', 
                  'Domestic_Box_Office_USD', 'Domestic_Gross_USD'], axis=1)

# Define features and target variable
X = data.drop('Worldwide_Gross_USD', axis=1)
y = data['Worldwide_Gross_USD']

# Check for missing values and fill them
print("Missing values per column:\n", X.isna().sum())
X.fillna(0, inplace=True)

# One-hot encode categorical features
X = pd.get_dummies(X)

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# --- Model Training ---
rf_regressor = RandomForestRegressor(n_estimators=50, random_state=42)
train_start = time.time()
rf_regressor.fit(X_train, y_train)
print(f"Initial model training completed in {time.time() - train_start:.2f} seconds")

# --- Model Evaluation ---
# Predictions
y_pred = rf_regressor.predict(X_test)

# Metrics on test data
mae_test = mean_absolute_error(y_test, y_pred)
mse_test = mean_squared_error(y_test, y_pred)
rmse_test = np.sqrt(mse_test)
r2_test = r2_score(y_test, y_pred)

print(f"Initial Model Performance on Test Set:")
print(f"  Mean Absolute Error (MAE): {mae_test:.2f}")
print(f"  Mean Squared Error (MSE): {mse_test:.2f}")
print(f"  Root Mean Squared Error (RMSE): {rmse_test:.2f}")
print(f"  R² Score: {r2_test:.2f}")

# --- Diagnostic Check ---
# Evaluate the model on training data
y_train_pred = rf_regressor.predict(X_train)

# Metrics on training data
mae_train = mean_absolute_error(y_train, y_train_pred)
mse_train = mean_squared_error(y_train, y_train_pred)
rmse_train = np.sqrt(mse_train)
r2_train = r2_score(y_train, y_train_pred)

print("\nDiagnostic Check (Training vs. Test Performance):")
print(f"Training Set Performance:")
print(f"  Mean Absolute Error (MAE): {mae_train:.2f}")
print(f"  Mean Squared Error (MSE): {mse_train:.2f}")
print(f"  Root Mean Squared Error (RMSE): {rmse_train:.2f}")
print(f"  R² Score: {r2_train:.2f}")

print("\nTest Set Performance (already shown above):")
print(f"  R² Score on Test Set: {r2_test:.2f}")

# --- Feature Importance Analysis ---
# Get feature importances
feature_importances = rf_regressor.feature_importances_
feature_names = X.columns

# Create a DataFrame for visualization
importance_df = pd.DataFrame({
    'Feature': feature_names,
    'Importance': feature_importances
}).sort_values(by='Importance', ascending=False)

# Display top features
print("\nTop 10 Features by Importance:")
print(importance_df.head(10))

# Plot feature importances
plt.figure(figsize=(12, 8))
sns.barplot(x='Importance', y='Feature', data=importance_df.head(20))
plt.title('Feature Importance in Random Forest Regression')
plt.xlabel('Importance Score')
plt.ylabel('Top Features')
plt.tight_layout()
plt.show()

# --- Hyperparameter Tuning ---
param_dist = {
    'n_estimators': [50, 100, 150],
    'max_depth': [10, 20, 30, None],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4],
    'max_features': ['sqrt', 'log2']
}

random_search = RandomizedSearchCV(
    estimator=rf_regressor,
    param_distributions=param_dist,
    n_iter=20,
    cv=3,
    scoring='neg_mean_squared_error',
    n_jobs=-1,
    verbose=2,
    random_state=42
)

# Fit the search
tuning_start = time.time()
random_search.fit(X_train, y_train)
print(f"Hyperparameter tuning completed in {time.time() - tuning_start:.2f} seconds")

# Display the best parameters found
print("Best parameters found:", random_search.best_params_)

# --- Re-train model with best parameters ---
best_rf = random_search.best_estimator_
y_pred_best = best_rf.predict(X_test)

# Evaluate best model on test data
mae_best = mean_absolute_error(y_test, y_pred_best)
mse_best = mean_squared_error(y_test, y_pred_best)
rmse_best = np.sqrt(mse_best)
r2_best = r2_score(y_test, y_pred_best)

print("\nOptimized Model Performance on Test Set:")
print(f"  Mean Absolute Error (MAE): {mae_best:.2f}")
print(f"  Mean Squared Error (MSE): {mse_best:.2f}")
print(f"  Root Mean Squared Error (RMSE): {rmse_best:.2f}")
print(f"  R² Score: {r2_best:.2f}")

# --- Display total execution time ---
print(f"\nTotal execution time: {time.time() - start_time:.2f} seconds")
