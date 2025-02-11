import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
import xgboost as xgb
import matplotlib.pyplot as plt
import shap

df = pd.read_csv(r"C:\Users\matas\OneDrive\Desktop\movie_project\Top Movies (Cleaned Data).csv")

df = df.drop(['id', 'Movie URL'], axis=1)
df = df.dropna()

df['Release Date'] = pd.to_datetime(df['Release Date'])
df['Release Year'] = df['Release Date'].dt.year
df['Release Month'] = df['Release Date'].dt.month
df['Release Day'] = df['Release Date'].dt.day
df = df.drop('Release Date', axis=1)

object_columns = df.select_dtypes(include=['object']).columns
print("Object columns detected:", object_columns)

for col in object_columns:
    if col in ['Video Release']:
        df[col] = pd.to_datetime(df[col], errors='coerce').astype('int64', errors='ignore')
    else:
        df[col] = df[col].astype('category').cat.codes

df = df.fillna(0)

categorical_columns = ['MPAA Rating', 'Franchise', 'Keywords', 'Source', 
                       'Production Method', 'Creative Type', 'Production/Financing Companies', 
                       'Production Countries', 'Languages']

for col in categorical_columns:
    if col in df.columns:
        df[col] = df[col].astype('category').cat.codes

df = pd.get_dummies(df, columns=['Genre'], prefix='Genre', drop_first=True)

target = 'Worldwide Gross (USD)'
X = df.drop(target, axis=1)
y = df[target]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

X_train = X_train.apply(pd.to_numeric, errors='coerce').fillna(0)
X_test = X_test.apply(pd.to_numeric, errors='coerce').fillna(0)

dtrain = xgb.DMatrix(X_train, label=y_train)
dtest = xgb.DMatrix(X_test, label=y_test)

params = {
    'objective': 'reg:squarederror',
    'max_depth': 6,
    'learning_rate': 0.05,
    'subsample': 0.8,
    'colsample_bytree': 0.8,
    'seed': 42
}

xgb_model = xgb.train(
    params=params,
    dtrain=dtrain,
    num_boost_round=1000,
    evals=[(dtest, 'eval')],
    early_stopping_rounds=10,
    verbose_eval=True
)

y_pred = xgb_model.predict(dtest)

rmse = np.sqrt(mean_squared_error(y_test, y_pred))
print(f"RMSE: {rmse}")

xgb.plot_importance(xgb_model)
plt.show()

X_train_df = pd.DataFrame(X_train, columns=X.columns)
X_test_df = pd.DataFrame(X_test, columns=X.columns)

bool_columns = X_train_df.select_dtypes(include=['bool']).columns
X_train_df[bool_columns] = X_train_df[bool_columns].astype(int)
X_test_df[bool_columns] = X_test_df[bool_columns].astype(int)

print("SHAP DataFrame types:", X_train_df.dtypes)
print("Null values in SHAP DataFrame:", X_train_df.isna().sum().sum())

explainer = shap.Explainer(xgb_model, X_train_df)

shap_values = explainer(X_test_df)

shap.summary_plot(shap_values, X_test_df)
