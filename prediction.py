import pandas as pd
import requests

# Carregar os dados
data = pd.read_csv('data/stocks_data.csv')
data = data[['Close', 'High', 'Low', 'Open', 'Volume']].values.tolist()

# Enviar os últimos 20 dias para a API
url = "http://localhost:5000/predict"
response = requests.post(url, json={"data": data[-20:]})

# Verificar a resposta
if response.status_code == 200:
    print("Previsão:", response.json())
else:
    print("Erro:", response.status_code, response.text)
