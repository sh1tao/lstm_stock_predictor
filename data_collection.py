import yfinance as yf
import pandas as pd

# Lista das ações a serem analisadas
symbols = ['TSLA']

# Datas de início e fim
start_date = '2010-01-01'
end_date = '2025-02-13'

# Baixar os dados do Yahoo Finance
df = yf.download(symbols, start=start_date, end=end_date)

# Converter para formato long
df = df.stack(level=1).reset_index()
df.columns = ['Date', 'Symbol', 'Close', 'High', 'Low', 'Open', 'Volume']
df.drop(columns=['Symbol'], inplace=True)

#  **1. Média Móvel Simples (SMA)**
df['SMA_10'] = df['Close'].rolling(window=10).mean()
df['SMA_50'] = df['Close'].rolling(window=50).mean()

#  **2. Média Móvel Exponencial (EMA)**
df['EMA_10'] = df['Close'].ewm(span=10, adjust=False).mean()
df['EMA_50'] = df['Close'].ewm(span=50, adjust=False).mean()

#  **3. Índice de Força Relativa (RSI)**
def compute_rsi(data, window=14):
    delta = data.diff(1)
    gain = (delta.where(delta > 0, 0)).rolling(window=window).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=window).mean()
    rs = gain / loss
    return 100 - (100 / (1 + rs))

df['RSI'] = compute_rsi(df['Close'])

#  **4. MACD (Média Móvel Convergência/Divergência)**
df['MACD'] = df['EMA_12'] = df['Close'].ewm(span=12, adjust=False).mean() - df['Close'].ewm(span=26, adjust=False).mean()
df['MACD_Signal'] = df['MACD'].ewm(span=9, adjust=False).mean()

#  **5. Bollinger Bands (BB)**
df['BB_Mid'] = df['Close'].rolling(window=20).mean()
df['BB_Upper'] = df['BB_Mid'] + (df['Close'].rolling(window=20).std() * 2)
df['BB_Lower'] = df['BB_Mid'] - (df['Close'].rolling(window=20).std() * 2)

#  **6. Estocástico (%K e %D)**
df['L14'] = df['Low'].rolling(window=14).min()
df['H14'] = df['High'].rolling(window=14).max()
df['%K'] = 100 * (df['Close'] - df['L14']) / (df['H14'] - df['L14'])
df['%D'] = df['%K'].rolling(window=3).mean()

# Remover valores NaN
df.dropna(inplace=True)

# Salvar os dados em CSV
df.to_csv('data/stocks_data.csv', index=False)
print(" Dados coletados e salvos em 'stocks_data.csv' com novos indicadores técnicos!")