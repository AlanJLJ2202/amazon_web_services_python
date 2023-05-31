from flask import Flask, render_template, request
import pandas as pd
import numpy as np
from sklearn.svm import OneClassSVM
from statsmodels.tsa.arima.model import ARIMA



import matplotlib
matplotlib.use('Agg')  # Establecer el backend de Matplotlib en 'Agg'

import matplotlib.pyplot as plt


import io
import base64

app = Flask(__name__)

@app.route('/', methods=['GET', 'POST'])
def home():
    if request.method == 'POST':
        
        # Predicciones de la serie de tiempo ARIMA
        time_series = data['Close']
        model = ARIMA(time_series, order=(1, 0, 1))
        model_fit = model.fit()
        predictions = model_fit.predict(start=len(time_series), end=len(time_series)+5)

        plt.figure(figsize=(10, 6))
        plt.plot(range(len(data), len(data) + len(predictions)), predictions, label='Predicciones')

        # Agregar etiquetas a las predicciones por día
        for i, pred in enumerate(predictions):
            plt.text(len(data) + i, pred, f'{pred:.2f}', ha='center', va='bottom')

        # Guardar el gráfico de predicciones en un objeto BytesIO
        only_predictions_img = io.BytesIO()
        plt.savefig(only_predictions_img, format='png')
        only_predictions_img.seek(0)

        # Generar el código base64 de la imagen de predicciones
        only_predictions_img_base64 = base64.b64encode(only_predictions_img.getvalue()).decode()


        plt.figure(figsize=(10, 6))
        plt.plot(range(len(data)), data['Close'], label='Datos')
        plt.plot(range(len(data), len(data) + len(predictions)), predictions, label='Predicciones')

        # Guardar el gráfico de predicciones en un objeto BytesIO
        predictions_img = io.BytesIO()
        plt.savefig(predictions_img, format='png')
        predictions_img.seek(0)

        # Generar el código base64 de la imagen de predicciones
        predictions_img_base64 = base64.b64encode(predictions_img.getvalue()).decode()


        #------------------------------------------------------------------------------------------------------

        # Obtener el archivo cargado desde el formulario
        archivo = request.files['archivo']

        umbral = float(request.form['nu'])

        # Leer los datos del archivo CSV
        data = pd.read_csv(archivo)

        # Seleccionar las columnas de interés
        #X = data[['start_price', 'end_price', 'maximum_price', 'minimum_price']]
        X = data['Close'].values.reshape(-1, 1)

        # Crear el modelo de detección de anomalías (One-Class SVM)
        clf = OneClassSVM(nu=umbral, kernel='rbf', gamma='scale')

        # Entrenar el modelo con los datos normales
        clf.fit(X)

        y_pred = clf.predict(X)
        anomaly_indices = np.where(y_pred == -1)[0]
        anomaly_prices = data['Close'].iloc[anomaly_indices]

        indices = np.arange(len(data))  # Crear un rango de índices

        plt.figure(figsize=(10, 6))
        plt.plot(indices, X, label='Datos')
        plt.plot(anomaly_indices, anomaly_prices, 'ro', label='Anomalías')
        plt.xlabel('Índice')
        plt.ylabel('Precio de Cierre (EUR)')
        plt.title('Detección de Anomalías')
        plt.legend()

        # Guardar el gráfico en un objeto BytesIO
        img = io.BytesIO()
        plt.savefig(img, format='png')
        img.seek(0)

        # Generar el código base64 de la imagen
        anomaly_img_base64 = base64.b64encode(img.getvalue()).decode()

        
        # Renderizar el template HTML y pasar los datos y las imágenes a la vista
        return render_template('result.html',
                                anomaly_prices=anomaly_prices, 
                                anomaly_img_base64=anomaly_img_base64, 
                                predictions_img_base64=predictions_img_base64, 
                                only_predictions_img_base64=only_predictions_img_base64)

    # Si la solicitud es GET, renderizar el formulario de carga de archivo
    return render_template('index.html')

if __name__ == '__main__':
    app.run(debug=True)