import boto3
import pandas as pd
import numpy as np
from io import StringIO, BytesIO
from datetime import datetime, timedelta
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential
from keras.layers import Dense, LSTM

class ETL:
    def extract(self):
        pass

    def transform_report(self):
        pass

    def load_report(self):
        pass

class AdaptationLayer():
    
    def read_csv_to_df(self, bucket, objects):
    
        #Get the columns of the objects
        csv_obj_init = bucket.Object(key=objects[0].key).get().get('Body').read().decode('utf-8')
        data = StringIO(csv_obj_init)
        df_init = pd.read_csv(data, delimiter=',')
        #Create a dataframe with the columns of the objects
        df_all = pd.DataFrame(columns=df_init.columns)
        #Concat the objects to the dataframe with the columns
        for obj in objects:
            csv_obj = bucket.Object(key=obj.key).get().get('Body').read().decode('utf-8')
            data = StringIO(csv_obj)
            df = pd.read_csv(data, delimiter=',')
            df_all = pd.concat([df,df_all], ignore_index=True)
        
        return df_all

    def write_df_to_s3(self, df_all, key, bucket_target):
    
        #Create the buffer to store the dataframe
        out_buffer = BytesIO()
        #Create a .parquet file
        df_all.to_parquet(out_buffer, index=False)
        #Upload the file to the bucket with the key and the .parquet file stored in the buffer
        bucket_target.put_object(Body=out_buffer.getvalue(), Key=key)
        pass

    def return_objects(self, bucket, arg_date_dt):
        #Get all the objects according to the condition given and return them
        objects = [obj for obj in bucket.objects.all() if datetime.strptime(obj.key.split("/")[0], '%Y-%m-%d').date() >= arg_date_dt]
        return objects

class ETL_S3(ETL):
    def __init__(self, bucket_name, bucket_target_name, arg_date):
        self.__s3 = boto3.resource('s3')
        self.__bucket = self.__s3.Bucket(bucket_name)
        self.__bucket_target = self.__s3.Bucket(bucket_target_name)
        self.arg_date_dt = datetime.strptime(arg_date, '%Y-%m-%d').date() - timedelta(days=1)
        self.__ap = AdaptationLayer()
    
    def extract(self):
        print("extract")
        objects = self.__ap.return_objects(self.__bucket, self.arg_date_dt)

        df_all = self.__ap.read_csv_to_df(self.__bucket, objects)
        print("fin extract")
        return df_all
    
    def transform(self):
        pass
    
    def load_report(self, df_all):
        print("load")
        #Generate a key to save the dataframe
        key = 'xetra_daily_report_' + datetime.today().strftime("%Y%m%d_%H%M%S") + '.parquet'
        
        #Write the file in the cloud
        self.__ap.write_df_to_s3(df_all, key, self.__bucket_target)
        print("end load")
        pass

    def etl_report(self):
        print("etl")
        target_objects = [obj for obj in self.__bucket_target.objects.all()]
        prq_obj = self.__bucket_target.Object(key=target_objects[-1].key).get().get('Body').read()
        data = BytesIO(prq_obj)
        df_report = pd.read_parquet(data)
        
        return df_report

class ApplicationLayer(ETL_S3):
    
    def transform_report(self, df_all):
        print("transform")
        df_all.dropna(inplace=True)

        df_all['start_price'] = df_all.sort_values(by=['Time']).groupby(['ISIN', 'Date'])['StartPrice'].transform('first')

        df_all['end_price'] = df_all.sort_values(by=['Time']).groupby(['ISIN', 'Date'])['EndPrice'].transform('last')

        df_all = df_all.query('"08:00" < Time < "12:00"').groupby(['ISIN', 'Date'], as_index=False).agg(start_price=('start_price', 'min'), end_price=('end_price', 'min'), minimum_price=('MinPrice', 'min'), maximum_price=('MaxPrice', 'max'), daily_traded_volume=('TradedVolume', 'sum'))
        
        df_all["end_price_mx"] = df_all["end_price"] * 19.08
        
        deviation = ['start_price','end_price']

        df_all["standard_deviation"] = df_all[deviation].std(axis=1)
        
        print("fin transform")
        return df_all
    

class NeuralNetwork():
    #def __init__(self, dataframe):
        #self.dataframe = dataframe

    
    def run(self):

        # Seleccionar el activo que se desea predecir
        activo = 'AT000000STR1'

        df = pd.read_csv('transformed_data.csv')

        df_activo = df[df['ISIN'] == activo].reset_index(drop=True)
        # Seleccionar columnas relevantes
        df_activo = df_activo[['Date', 'end_price']]
        # Establecer fecha como índice
        df_activo.set_index('Date', inplace=True)
        # Escalar datos
        scaler = MinMaxScaler(feature_range=(0, 1))
        scaled_data = scaler.fit_transform(df_activo)
        # Dividir datos en entrenamiento y prueba
        train_data = scaled_data[:int(len(df_activo)*0.8), :]
        test_data = scaled_data[int(len(df_activo)*0.8):, :]

        # Dividir datos en entradas y etiquetas
        def create_dataset(data, time_step=1):
            X, Y = [], []
            for i in range(len(data)-time_step-1):
                a = data[i:(i+time_step), 0]
                X.append(a)
                Y.append(data[i+time_step, 0])
            return np.array(X), np.array(Y)

        time_step = 10
        X_train, Y_train = create_dataset(train_data, time_step)
        X_test, Y_test = create_dataset(test_data, time_step)

        # Reestructurar datos para LSTM
        X_train = X_train.reshape(X_train.shape[0], time_step, 1)
        X_test = X_test.reshape(X_test.shape[0], X_test.shape[1], 1)

        # Crear modelo LSTM
        model = Sequential()
        model.add(LSTM(50, return_sequences=True, input_shape=(X_train.shape[1], 1)))
        model.add(LSTM(50, return_sequences=True))
        model.add(LSTM(50))
        model.add(Dense(1))
        model.compile(loss='mean_squared_error', optimizer='adam')

        # Entrenar modelo
        model.fit(X_train, Y_train, validation_data=(X_test, Y_test), epochs=100, batch_size=64, verbose=1)

        # Realizar predicciones
        train_predict = model.predict(X_train)
        test_predict = model.predict(X_test)

        # Invertir escala de datos para obtener valores reales
        train_predict = scaler.inverse_transform(train_predict)
        Y_train = scaler.inverse_transform([Y_train])
        test_predict = scaler.inverse_transform(test_predict)
        Y_test = scaler.inverse_transform([Y_test])

        # Calcular errores de entrenamiento y prueba
        train_error = np.sqrt(np.mean(np.power((Y_train-train_predict), 2)))
        test_error = np.sqrt(np.mean(np.power((Y_test-test_predict), 2)))

        print("Error de entrenamiento:", train_error)
        print("Error de prueba:", test_error)

        # Hacer predicciones a una semana
        x_input = test_data[-time_step:, 0]
        x_input = x_input.reshape(1, -1)
        x_input = scaler.transform(x_input)
        temp_input = list(x_input)
        temp_input = temp_input[0].tolist()

        prediccion = []
        dias_a_predecir = 7

        for i in range(dias_a_predecir):
            if len(temp_input) > time_step:
                temp_input = temp_input[1:]
            x_input = np.array(temp_input)
            x_input = x_input.reshape(1, -1)
            x_input = x_input.reshape((1, time_step, 1))
            yhat = model.predict(x_input, verbose=0)
            temp_input.append(yhat[0, 0])
            prediccion.append(yhat[0, 0])

        # Escalar de vuelta los valores de predicción
        prediccion = np.array(prediccion)
        prediccion = prediccion.reshape(-1, 1)
        prediccion = scaler.inverse_transform(prediccion)

        # Imprimir las predicciones
        print("Predicciones para la próxima semana:")
        for i in range(len(prediccion)):
            print(f"Día {i+1}: {prediccion[i][0]:.2f}")

    
    


    
#al = ApplicationLayer(arg_date='2022-12-31', bucket_name='xetra-1234', bucket_target_name='xetra-ajlj')
#df = al.extract()
#print(df)
#transformed_data = al.transform_report(df_all=df)
#print(transformed_data)

#transformed_data.to_csv('transformed_data.csv', index=False)

#al.load_report(df_all=transformed_data)
#report = al.etl_report()
#print(report)

neuronal_network_test = NeuralNetwork()
print(neuronal_network_test.run())
