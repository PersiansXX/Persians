import ccxt
import pandas as pd
import numpy as np
import talib
import time
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler

api_key = 'YOUR_BINANCE_API_KEY'
api_secret = 'YOUR_BINANCE_API_SECRET'

exchange = ccxt.binance({
    'apiKey': api_key,
    'secret': api_secret,
})

symbols = ['BTC/USDT', 'ETH/USDT', 'BNB/USDT']
amounts = {'BTC/USDT': 0.001, 'ETH/USDT': 0.01, 'BNB/USDT': 0.1}
timeframe = '5m'
trade_log_file = 'trade_log.csv'

def fetch_data(symbol, timeframe, limit=100):
    bars = exchange.fetch_ohlcv(symbol, timeframe, limit=limit)
    df = pd.DataFrame(bars, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
    return df

def apply_technical_indicators(df):
    df['SMA'] = talib.SMA(df['close'], timeperiod=20)
    df['EMA'] = talib.EMA(df['close'], timeperiod=20)
    df['RSI'] = talib.RSI(df['close'], timeperiod=14)
    df['Momentum'] = talib.MOM(df['close'], timeperiod=10)
    df['Returns'] = df['close'].pct_change()
    return df.dropna()

def train_model(df):
    features = df[['SMA', 'EMA', 'RSI', 'Momentum']]
    target = np.where(df['Returns'].shift(-1) > 0, 1, 0)
    scaler = StandardScaler()
    features = scaler.fit_transform(features)
    model = RandomForestClassifier()
    model.fit(features, target)
    return model, scaler

def make_decision(df, model, scaler):
    features = df[['SMA', 'EMA', 'RSI', 'Momentum']]
    features = scaler.transform(features)
    prediction = model.predict(features[-1].reshape(1, -1))
    last_close = df['close'].iloc[-1]
    last_rsi = df['RSI'].iloc[-1]
    
    if prediction == 1 and last_rsi < 70:
        return 'buy'
    elif last_rsi > 70:
        return 'sell'
    else:
        return 'hold'

def record_trade(symbol, decision, amount, price):
    trade = {
        'symbol': symbol,
        'decision': decision,
        'amount': amount,
        'price': price,
        'timestamp': time.time()
    }
    trade_df = pd.DataFrame([trade])
    trade_df.to_csv(trade_log_file, mode='a', header=not pd.read_csv(trade_log_file).empty, index=False)

def calculate_profit():
    trades = pd.read_csv(trade_log_file)
    profit = 0
    for _, trade in trades.iterrows():
        if trade['decision'] == 'buy':
            profit -= trade['amount'] * trade['price']
        elif trade['decision'] == 'sell':
            profit += trade['amount'] * trade['price']
    return profit

def main():
    model = None
    scaler = None
    while True:
        for symbol in symbols:
            df = fetch_data(symbol, timeframe)
            df = apply_technical_indicators(df)
            
            if model is None or scaler is None:
                model, scaler = train_model(df)
            
            decision = make_decision(df, model, scaler)
            
            balance = exchange.fetch_balance()
            usdt_balance = balance['total']['USDT']
            coin_balance = balance['total'][symbol.split('/')[0]]
            amount = amounts[symbol]
            
            if decision == 'buy' and usdt_balance >= df['close'].iloc[-1] * amount:
                order = exchange.create_market_buy_order(symbol, amount)
                record_trade(symbol, 'buy', amount, df['close'].iloc[-1])
                print(f"Bought {amount} of {symbol} at {df['close'].iloc[-1]} USDT")
            elif decision == 'sell' and coin_balance >= amount:
                order = exchange.create_market_sell_order(symbol, amount)
                record_trade(symbol, 'sell', amount, df['close'].iloc[-1])
                print(f"Sold {amount} of {symbol} at {df['close'].iloc[-1]} USDT")
        
        profit = calculate_profit()
        print(f"Current Profit: {profit} USDT")
        
        time.sleep(20)

if __name__ == "__main__":
    main()