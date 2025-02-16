from flask import Flask, request, jsonify
from flask_monitoringdashboard import bind
import torch
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from waitress import serve

# Configuração do Flask
app = Flask(__name__)

# Configura o Flask-MonitoringDashboard
bind(app)

# Carregar o modelo
class LSTM(torch.nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, output_size):
        super(LSTM, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.lstm = torch.nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = torch.nn.Linear(hidden_size, output_size)

    def forward(self, x):
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        out, _ = self.lstm(x, (h0, c0))
        out = self.fc(out[:, -1, :])
        return out

# Parâmetros do modelo
input_size = 5
hidden_size = 50
num_layers = 2
output_size = 5

# Instanciar o modelo
model = LSTM(input_size, hidden_size, num_layers, output_size)
model.load_state_dict(torch.load('lstm_stock_model.pth'))
model.eval()  # Colocar o modelo em modo de avaliação

# Carregar o scaler (se necessário)
scaler = MinMaxScaler(feature_range=(0, 1))

# Rota da API para previsões
@app.route('/predict', methods=['POST'])
def predict():
    """
    Faz a previsão dos preços futuros com base nos dados históricos.
    ---
    tags:
      - Previsão
    summary: Prever preços futuros
    description: |
      Este endpoint recebe dados históricos de preços de ações e retorna a previsão dos preços para os próximos dias.
      Os dados devem ser fornecidos em formato JSON, contendo uma lista de listas com os valores de 'Close', 'High', 'Low', 'Open' e 'Volume'.
    operationId: predictPrices
    consumes:
      - application/json
    produces:
      - application/json
    parameters:
      - in: body
        name: body
        required: true
        description: Dados históricos para previsão
        schema:
          type: object
          properties:
            data:
              type: array
              items:
                type: array
                items:
                  type: number
                example: [1.0, 2.0, 3.0, 4.0, 5.0]
              example:
                - [1.0, 2.0, 3.0, 4.0, 5.0]
                - [1.1, 2.1, 3.1, 4.1, 5.1]
                - [1.2, 2.2, 3.2, 4.2, 5.2]
    responses:
      200:
        description: Previsão dos preços futuros
        schema:
          type: object
          properties:
            prediction:
              type: array
              items:
                type: number
              example: [100.5, 101.2, 102.0]
      400:
        description: Erro na requisição
        schema:
          type: object
          properties:
            error:
              type: string
              example: "Erro ao processar os dados."
    """
    try:
        # Receber os dados da requisição
        data = request.json['data']  # Espera-se uma lista de listas com os dados históricos
        data = np.array(data)

        # Normalizar os dados (se necessário)
        scaled_data = scaler.fit_transform(data)

        # Converter para tensor
        input_tensor = torch.tensor(scaled_data, dtype=torch.float32).unsqueeze(0)

        # Fazer a previsão
        with torch.no_grad():
            prediction = model(input_tensor).cpu().numpy()

        # Desnormalizar a previsão (se necessário)
        prediction_denormalized = scaler.inverse_transform(prediction)

        # Retornar a previsão como JSON
        return jsonify({"prediction": prediction_denormalized.tolist()})
    except Exception as e:
        return jsonify({"error": str(e)}), 400

# Função para iniciar o servidor com Waitress
def start_server():
    print("Iniciando servidor Waitress...")
    serve(app, host='0.0.0.0', port=5000)  # Usar Waitress para servir a aplicação

# Ponto de entrada
if __name__ == '__main__':
    start_server()