import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.linear_model import BayesianRidge
from lightgbm import LGBMRegressor
from xgboost import XGBRegressor
from catboost import CatBoostRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.cluster import KMeans

df = pd.read_csv('/content/drive/MyDrive/LPFB/Dataset/Train_data.csv')
df['DATE'] = pd.to_datetime(df['DATE'])

def fet_eng(df):
    df['temp_diff'] = df['T2M'] - df['T2MDEW']
    df['humidity_ratio'] = df['QV2M'] / (df['RH2M'] + 1e-3)
    df['dayofyear'] = df['DATE'].dt.dayofyear
    df['month'] = df['DATE'].dt.month
    df['week'] = df['DATE'].dt.isocalendar().week.astype(int)
    df['season'] = df['month'] % 12 // 3 + 1
    df['sin_doy'] = np.sin(2 * np.pi * df['dayofyear'] / 365)
    df['cos_doy'] = np.cos(2 * np.pi * df['dayofyear'] / 365)
    df['wetness_spread'] = df['T2MWET'] - df['T2MDEW']
    df['pressure_drop'] = 96.3 - df['PS']
    df['wind_humidity_mix'] = df['WS2M'] * df['RH2M']
    df['temp_dew_ratio'] = df['T2M'] / (df['T2MDEW'] + 1e-3)
    df['dew_temp_interact'] = df['T2MDEW'] * df['RH2M']
    df['wind_pressure'] = df['WS2M'] * df['PS']
    df['climate_cluster'] = KMeans(n_clusters=4, random_state=42).fit_predict(
        df[['T2M', 'RH2M', 'QV2M', 'WS2M', 'PS']]
    )
    return df

df = fet_eng(df)

x = df.drop(columns=["ID", "DATE", "Target"])
y_log = np.log1p(df["Target"])
x_tra, x_val, y_tra, y_val = train_test_split(x, y_log, test_size=0.2, random_state=42)
tr_val = np.expm1(y_val)

xgb = XGBRegressor(
    n_estimators=500, max_depth=8, learning_rate=0.03,
    subsample=0.85, colsample_bytree=0.85, reg_lambda=1.2, gamma=0.3,
    random_state=42, n_jobs=-1
).fit(x_tra, y_tra)

lgb = LGBMRegressor(
    n_estimators=600, max_depth=8, learning_rate=0.03,
    subsample=0.85, colsample_bytree=0.85, reg_lambda=1.2,
    min_data_in_leaf=5, random_state=42, n_jobs=-1
).fit(x_tra, y_tra)

cat = CatBoostRegressor(
    iterations=600, depth=8, learning_rate=0.03,
    subsample=0.85, reg_lambda=1.2, verbose=0,
    bagging_temperature=0.8, random_state=42
).fit(x_tra, y_tra)

rf = RandomForestRegressor(
    n_estimators=400, max_depth=12, random_state=42, n_jobs=-1
).fit(x_tra, y_tra)

xgb_pred = np.expm1(xgb.predict(x_val))
lgb_pred = np.expm1(lgb.predict(x_val))
cat_pred = np.expm1(cat.predict(x_val))
rf_pred  = np.expm1(rf.predict(x_val))

xgb_pred = np.clip(xgb_pred, 0, 60)
lgb_pred = np.clip(lgb_pred, 0, 60)
cat_pred = np.clip(cat_pred, 0, 60)
rf_pred  = np.clip(rf_pred,  0, 60)

meta_X = np.column_stack([xgb_pred, lgb_pred, cat_pred, rf_pred])
me_mod = BayesianRidge()
me_mod.fit(meta_X, tr_val)
ens_pred = me_mod.predict(meta_X)

fi_pred = np.clip(ens_pred, 0, 60)

def rmse(y, yhat): return np.sqrt(mean_squared_error(y, yhat))

print(f"XGBoost RMSE      : {rmse(tr_val, xgb_pred):.6f}")
print(f"LightGBM RMSE     : {rmse(tr_val, lgb_pred):.6f}")
print(f"CatBoost RMSE     : {rmse(tr_val, cat_pred):.6f}")
print(f"RandomForest RMSE : {rmse(tr_val, rf_pred):.6f}")
print(f"Ensemble RMSE     : {rmse(tr_val, fi_pred):.6f}")
