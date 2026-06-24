"""
train_model.py — Stock Price Prediction
AI/ML Internship Project
Run this script once to train and save all models.
"""
import pandas as pd
import numpy as np
import joblib
import warnings
warnings.filterwarnings('ignore')

from sklearn.preprocessing import MinMaxScaler
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score

# ── 1. Load Data ─────────────────────────────────────────────────────────────
print("Loading stock data...")
try:
    import yfinance as yf
    df = yf.download('AAPL', start='2018-01-01', end='2024-12-31', progress=False)
    df = df.reset_index()
    print(f"Downloaded {len(df)} rows from Yahoo Finance")
except Exception as e:
    print(f"Yahoo Finance unavailable ({e}), using CSV...")
    df = pd.read_csv('AAPL_stock_data.csv')

# ── 2. Feature Engineering ───────────────────────────────────────────────────
print("Engineering features...")
df['MA_10']        = df['Close'].rolling(10).mean()
df['MA_30']        = df['Close'].rolling(30).mean()
df['MA_50']        = df['Close'].rolling(50).mean()
df['Daily_Return'] = df['Close'].pct_change()
df['Volatility']   = df['Daily_Return'].rolling(10).std()
df['Price_Range']  = df['High'] - df['Low']

# Lag features
for lag in [1, 2, 3, 5]:
    df[f'Close_lag{lag}'] = df['Close'].shift(lag)

# Target: next day close
df['Target'] = df['Close'].shift(-1)
df.dropna(inplace=True)
df.to_csv('AAPL_stock_data.csv', index=False)
print(f"Saved {len(df)} rows to AAPL_stock_data.csv")

# ── 3. Prepare ML Inputs ─────────────────────────────────────────────────────
features = ['Open','High','Low','Close','Volume',
            'MA_10','MA_30','MA_50','Daily_Return',
            'Volatility','Price_Range',
            'Close_lag1','Close_lag2','Close_lag3','Close_lag5']

X = df[features]
y = df['Target']

# Scale features 0-1
scaler = MinMaxScaler()
X_scaled = scaler.fit_transform(X)

# Time-series split (NO shuffle)
split    = int(len(X_scaled) * 0.8)
X_train  = X_scaled[:split];  X_test = X_scaled[split:]
y_train  = y.iloc[:split];    y_test = y.iloc[split:]
print(f"Train: {len(X_train)} | Test: {len(X_test)}")

# ── 4. Train Linear Regression ───────────────────────────────────────────────
print("\nTraining Linear Regression...")
lr = LinearRegression()
lr.fit(X_train, y_train)
lr_pred = lr.predict(X_test)
print(f"  R2  : {r2_score(y_test, lr_pred):.4f}")
print(f"  RMSE: {np.sqrt(mean_squared_error(y_test, lr_pred)):.4f}")

# ── 5. Train Random Forest ───────────────────────────────────────────────────
print("\nTraining Random Forest (200 trees)...")
rf = RandomForestRegressor(n_estimators=200, max_depth=15, random_state=42)
rf.fit(X_train, y_train)
rf_pred = rf.predict(X_test)
print(f"  R2  : {r2_score(y_test, rf_pred):.4f}")
print(f"  RMSE: {np.sqrt(mean_squared_error(y_test, rf_pred)):.4f}")

# ── 6. Save PKL Files ────────────────────────────────────────────────────────
print("\nSaving models...")
joblib.dump(lr,       'linear_model.pkl')
joblib.dump(rf,       'rf_model.pkl')
joblib.dump(scaler,   'scaler.pkl')
joblib.dump(features, 'features.pkl')
print("Done! Files saved: linear_model.pkl, rf_model.pkl, scaler.pkl, features.pkl")