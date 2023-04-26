import numpy as np

# Cargamos el dataframe desde el archivo CSV
import pandas as pd
df = pd.read_csv('transformed_data.csv')

# Preparamos los datos para el entrenamiento
X = df['end_price'].values[:-7]  # Tomamos todas las filas menos las últimas 7
y = df['end_price'].values[7:]   # Tomamos las filas desde la 7 en adelante
X = np.expand_dims(X, axis=1)    # Agregamos una dimensión adicional para que tenga la forma (n_samples, 1)
y = np.expand_dims(y, axis=1)

# Normalizamos los datos para facilitar el entrenamiento
from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler()
X_scaled = scaler.fit_transform(X)
y_scaled = scaler.fit_transform(y)

# Definimos la arquitectura de la red neuronal
class NeuralNetwork:
    def __init__(self):
        self.weights = np.random.rand(1, 4)  # Inicializamos los pesos aleatoriamente

    def sigmoid(self, x):
        return 1 / (1 + np.exp(-x))

    def sigmoid_derivative(self, x):
        return x * (1 - x)

    def train(self, X, y, epochs):
        for i in range(epochs):
            # Forward propagation
            z = np.dot(X, self.weights.T)
            output = self.sigmoid(z)

            # Backward propagation
            error = y - output
            d_weights = np.dot(error.T, X)
            self.weights += d_weights

    def predict(self, X):
        z = np.dot(X, self.weights.T)
        output = self.sigmoid(z)
        return output

# Entrenamos la red neuronal
nn = NeuralNetwork()
nn.train(X_scaled, y_scaled, epochs=1000)

# Hacemos la predicción para la próxima semana
last_week = df['end_price'].values[-7:]        # Tomamos las últimas 7 filas
last_week_scaled = scaler.fit_transform(last_week.reshape(-1, 1))
prediction_scaled = nn.predict(last_week_scaled)
prediction = scaler.inverse_transform(prediction_scaled)

print("La predicción para la próxima semana es:", prediction)