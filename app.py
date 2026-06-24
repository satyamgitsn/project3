from flask import Flask, render_template, request, jsonify
import joblib, numpy as np, pandas as pd, json

app = Flask(__name__)

# Load models
scaler = joblib.load('scaler.pkl')
lr_model = joblib.load('linear_model.pkl')
rf_model = joblib.load('rf_model.pkl')
features = joblib.load('features.pkl')

def prepare_input(data):
    close = float(data['close'])
    open_ = float(data['open'])
    high = float(data['high'])
    low = float(data['low'])
    volume = float(data['volume'])
    ma10 = float(data.get('ma10', close * 0.98))
    ma30 = float(data.get('ma30', close * 0.96))
    ma50 = float(data.get('ma50', close * 0.94))
    daily_return = (close - open_) / open_
    volatility = abs(daily_return) * 0.5
    price_range = high - low
    lag1 = float(data.get('lag1', close * 0.99))
    lag2 = float(data.get('lag2', close * 0.98))
    lag3 = float(data.get('lag3', close * 0.97))
    lag5 = float(data.get('lag5', close * 0.96))
    return np.array([[open_, high, low, close, volume, ma10, ma30, ma50,
                      daily_return, volatility, price_range, lag1, lag2, lag3, lag5]])

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.json
        X = prepare_input(data)
        X_scaled = scaler.transform(X)
        model_choice = data.get('model', 'linear')
        if model_choice == 'random_forest':
            pred = rf_model.predict(X_scaled)[0]
        else:
            pred = lr_model.predict(X_scaled)[0]
        close = float(data['close'])
        change = pred - close
        pct_change = (change / close) * 100
        return jsonify({
            'prediction': round(float(pred), 2),
            'current_price': round(close, 2),
            'change': round(float(change), 2),
            'pct_change': round(float(pct_change), 2),
            'signal': 'BUY' if change > 0 else 'SELL',
            'model': model_choice
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 400

@app.route('/history')
def history():
    df = pd.read_csv('AAPL_stock_data.csv')
    df = df.tail(90)[['Date', 'Close', 'MA_10', 'MA_30']].copy()
    return jsonify(df.to_dict(orient='records'))

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)