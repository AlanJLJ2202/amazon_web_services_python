import pandas as pd
import numpy as np
from statsmodels.tsa.arima.model import ARIMA
from sklearn.cluster import KMeans
from sklearn.ensemble import IsolationForest
from sklearn.svm import OneClassSVM


import matplotlib.pyplot as plt


# Cargar los datos del dataset de Xetra
data = pd.read_csv('Data/Bayer.csv')



''' PRIMERO
# Filtrar el dataset por las ISIN de las acciones deseadas
isin_list = ['AT000000STR1', 'AT00000FACC2', 'AT0000606306', 'AT0000730007', 'AT0000743059']  # Reemplaza con las ISIN de las acciones deseadas
filtered_data = data[data['ISIN'].isin(isin_list)]

# Crear una figura y ejes para el gráfico
fig, ax = plt.subplots()

# Iterar sobre las ISIN filtradas
for isin in isin_list:
    # Filtrar los datos para la ISIN actual
    time_series = filtered_data[filtered_data['ISIN'] == isin]['end_price']

    # Crear el modelo ARIMA y realizar predicciones
    model = ARIMA(time_series, order=(1, 0, 0))
    model_fit = model.fit()
    predictions = model_fit.predict(start=len(time_series), end=len(time_series) + 99)

    # Graficar los valores reales y las predicciones
    ax.plot(time_series.index, time_series.values, label=f'Valores reales - ISIN {isin}')
    ax.plot(predictions.index, predictions.values, label=f'Predicciones - ISIN {isin}')

# Configurar las etiquetas de los ejes y la leyenda
ax.set_xlabel('Índice temporal')
ax.set_ylabel('Precio de cierre')
ax.legend()

# Mostrar el gráfico
plt.show()
'''


# Seleccionar las columnas de interés
#X = data[['start_price', 'end_price', 'maximum_price', 'minimum_price']]
X = data['Close']


# Crear el modelo de detección de anomalías (One-Class SVM)
clf = OneClassSVM(nu=0.01, kernel='rbf', gamma='scale')

# Entrenar el modelo con los datos normales
clf.fit(X)

# Predecir las anomalías en los datos
y_pred = clf.predict(X)

# Obtener los índices de las anomalías
anomaly_indices = np.where(y_pred == -1)[0]


# Obtener el valor más alto de la columna "end_price"
max_end_price = data['end_price'].max()

print("El end_price más alto es:", max_end_price)

# Plotear los datos y las anomalías
plt.scatter(X.index, X['end_price'], label='Datos')
plt.scatter(X.index[anomaly_indices], X['end_price'].iloc[anomaly_indices], color='red', label='Anomalías')
plt.xlabel('Índice')
plt.ylabel('Precio de Cierre')
plt.title('Detección de Anomalías')
plt.legend()
plt.show()
