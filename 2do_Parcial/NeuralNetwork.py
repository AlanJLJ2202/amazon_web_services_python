import pandas as pd
from keras.models import Sequential
from keras.layers import Dense

# cargar los datos en un dataframe de pandas
df = pd.read_csv('transformed_data.csv')

# separar los datos de entrada y de salida
X = df[['start_price', 'minimum_price', 'maximum_price', 'daily_traded_volume']].values
y = df['end_price'].values

# definir el modelo
model = Sequential()
model.add(Dense(12, input_dim=4, activation='relu'))
model.add(Dense(8, activation='relu'))
model.add(Dense(1, activation='linear'))
model.compile(loss='mse', optimizer='adam', metrics=['mse'])

# entrenar el modelo
model.fit(X, y, epochs=10, batch_size=10)

# realizar una predicción con nuevos datos
nuevos_datos = pd.DataFrame({'start_price': [10], 'minimum_price': [9], 'maximum_price': [12], 'daily_traded_volume': [500]})
prediccion = model.predict(nuevos_datos)

print(prediccion)