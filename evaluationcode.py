import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error
import math
import mlflow
import mlflow.pytorch

# Set device configuration
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Hyperparameters
input_size = 5  # Number of features in the input data (e.g., 'Open', 'High', 'Low', 'Close', 'Volume')
hidden_size = 50
num_layers = 2
num_epochs = 100
batch_size = 64
learning_rate = 0.001
sequence_length = 20  # Length of the input sequences
future_days = 5  # Number of days to predict in the future
output_size = future_days  # Model will predict 'future_days' days ahead

# Load and preprocess the stock data
data = pd.read_csv('data/stocks_data.csv')
data = data[['Close', 'High', 'Low', 'Open', 'Volume']]  # Select relevant features
scaler = MinMaxScaler(feature_range=(0, 1))
scaled_data = scaler.fit_transform(data.values)  # Normalize the data


# Prepare the dataset for training and testing
def create_sequences_with_future(data, seq_length, future_days):
    sequences = []
    labels = []
    for i in range(len(data) - seq_length - future_days + 1):
        seq = data[i:i + seq_length]
        label = data[i + seq_length:i + seq_length + future_days, 0]  # Predicting the 'Close' price for future_days
        sequences.append(seq)
        labels.append(label)
    return np.array(sequences), np.array(labels)


X, y = create_sequences_with_future(scaled_data, sequence_length, future_days)

# Split the dataset into training and testing sets
train_size = int(len(X) * 0.8)
train_X, test_X = X[:train_size], X[train_size:]
train_y, test_y = y[:train_size], y[train_size:]

# Convert to torch tensors
train_X = torch.tensor(train_X, dtype=torch.float32).to(device)
train_y = torch.tensor(train_y, dtype=torch.float32).to(device)
test_X = torch.tensor(test_X, dtype=torch.float32).to(device)
test_y = torch.tensor(test_y, dtype=torch.float32).to(device)

# Create DataLoaders
train_dataset = TensorDataset(train_X, train_y)
test_dataset = TensorDataset(test_X, test_y)
train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=False)


# LSTM Model
class LSTM(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, output_size):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        out, _ = self.lstm(x, (h0, c0))
        out = self.fc(out[:, -1, :])
        return out


# Training the model
def train_model():
    model = LSTM(input_size, hidden_size, num_layers, output_size).to(device)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    mlflow.set_experiment("LSTM Stock Price Prediction")
    with mlflow.start_run():
        # Log model parameters
        mlflow.log_param("input_size", input_size)
        mlflow.log_param("hidden_size", hidden_size)
        mlflow.log_param("num_layers", num_layers)
        mlflow.log_param("output_size", output_size)
        mlflow.log_param("num_epochs", num_epochs)
        mlflow.log_param("batch_size", batch_size)
        mlflow.log_param("learning_rate", learning_rate)

        for epoch in range(num_epochs):
            model.train()
            running_loss = 0.0

            for i, (sequences, labels) in enumerate(train_loader):
                sequences, labels = sequences.to(device), labels.to(device)

                # Forward pass
                outputs = model(sequences)
                loss = criterion(outputs, labels)

                # Backward pass and optimization
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                running_loss += loss.item()

            print(f"Epoch [{epoch + 1}/{num_epochs}], Loss: {running_loss / len(train_loader):.4f}")

        # Save the model
        mlflow.pytorch.log_model(model, "lstm_stock_model")

        # Evaluate the model
        evaluate_model(model, criterion)


def evaluate_model(model, criterion):
    model.eval()
    test_loss = 0.0
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for sequences, labels in test_loader:
            sequences, labels = sequences.to(device), labels.to(device)
            outputs = model(sequences)
            loss = criterion(outputs, labels)
            test_loss += loss.item()

            all_preds.append(outputs.cpu().detach().numpy())  # Use .detach() aqui
            all_labels.append(labels.cpu().detach().numpy())  # Use .detach() aqui

    average_test_loss = test_loss / len(test_loader)
    print(f"Test Loss: {average_test_loss:.4f}")

    all_preds = np.concatenate(all_preds, axis=0)
    all_labels = np.concatenate(all_labels, axis=0)

    # Calculate metrics
    mae = mean_absolute_error(all_labels, all_preds)
    mse = mean_squared_error(all_labels, all_preds)
    rmse = math.sqrt(mse)

    print(f"MAE: {mae:.4f}")
    print(f"MSE: {mse:.4f}")
    print(f"RMSE: {rmse:.4f}")

    # Predict future days' closing prices
    next_day_input = test_X[-1].unsqueeze(0)  # Get the last sequence from test set
    future_predictions = model(next_day_input).cpu().detach().numpy()  # Use .detach() aqui

    # Denormalize the predictions
    future_predictions_denormalized = []
    for i in range(future_days):
        prediction = scaler.inverse_transform([[0, 0, 0, future_predictions[0, i], 0]])[0, 3]  # Reverse scaling
        future_predictions_denormalized.append(prediction)
        print(f"Day {i + 1} predicted 'Close' price: {prediction:.4f}")
    # Exemplo de reversão da normalização do MAE

    mae_original_scale = scaler.inverse_transform([[0, 0, 0, 0.0266, 0]])[0, 3]
    print(f"MAE na escala original: {mae_original_scale:.4f}")

    # Save the model to a file
    torch.save(model.state_dict(), 'lstm_stock_model.pth')
    print("Model saved to lstm_stock_model.pth")


# Run the training and evaluation
train_model()
