from flask import Flask, request, jsonify
from flask_swagger_ui import get_swaggerui_blueprint
from prometheus_flask_exporter import PrometheusMetrics
import torch
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from waitress import serve

# Configuração do Flask
app = Flask(__name__)

# Configuração do Prometheus
metrics = PrometheusMetrics(app)
metrics.info("lstm_stock_predictor", "API de Previsão de Ações", version="1.0")

# Configuração do Swagger-UI
SWAGGER_URL = "/api/docs"
API_URL = "/static/swagger.yaml"
swaggerui_blueprint = get_swaggerui_blueprint(SWAGGER_URL, API_URL)
app.register_blueprint(swaggerui_blueprint, url_prefix=SWAGGER_URL)


# Definição do modelo LSTM
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

# Carregar o modelo treinado
model = LSTM(input_size, hidden_size, num_layers, output_size)
model.load_state_dict(torch.load("lstm_stock_model.pth"))
model.eval()  # Colocar o modelo em modo de avaliação

# Inicializar o scaler
scaler = MinMaxScaler(feature_range=(0, 1))

# Métrica personalizada para contar requisições na rota /predict
predict_counter = metrics.counter(
    "predict_requests", "Número de requisições na rota /predict"
)


# Rota da API para previsões
@app.route("/predict", methods=["POST"])
@predict_counter
@metrics.summary("predict_latency", "Tempo de resposta da rota /predict")
def predict():
    """Faz a previsão dos preços futuros com base nos dados históricos."""
    try:
        # Receber os dados da requisição
        data = request.json["data"]  # Espera-se uma lista de listas com os dados históricos
        data = np.array(data)

        # Normalizar os dados
        scaled_data = scaler.fit_transform(data)

        # Converter para tensor
        input_tensor = torch.tensor(scaled_data, dtype=torch.float32).unsqueeze(0)

        # Fazer a previsão
        with torch.no_grad():
            prediction = model(input_tensor).cpu().numpy()

        # Desnormalizar a previsão
        prediction_denormalized = scaler.inverse_transform(prediction)

        # Retornar a previsão como JSON
        return jsonify({"prediction": prediction_denormalized.tolist()})

    except Exception as e:
        return jsonify({"error": str(e)}), 400


# Função para iniciar o servidor com Waitress
def start_server():
    print("Iniciando servidor Waitress...")
    serve(app, host="0.0.0.0", port=5000)  # Usar Waitress para servir a aplicação


# Ponto de entrada
if __name__ == "__main__":
    start_server()
