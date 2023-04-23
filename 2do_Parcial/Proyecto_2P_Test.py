import boto3
import pandas as pd
import numpy as np
from io import StringIO, BytesIO
from datetime import datetime, timedelta

class Extract():
    #Se crea el constructor de la clase
    def __init__(self, bucket_name, date):
        self.__s3 = boto3.resource('s3')
        self.__bucket = self.__s3.Bucket(bucket_name)
        self.date = datetime.strptime(date, '%Y-%m-%d').date() - timedelta(days=1)

    #Se crea el método para leer los archivos csv y convertirlos en un dataframe
    def read_csv_to_df(self, objects):
        csv_obj_init = self.__bucket.Object(key=objects[0].key).get().get('Body').read().decode('utf-8')
        data = StringIO(csv_obj_init)
        df_init = pd.read_csv(data, delimiter=',')
        df_all = pd.DataFrame(columns=df_init.columns)
        
        for obj in objects:
            csv_obj = self.__bucket.Object(key=obj.key).get().get('Body').read().decode('utf-8')
            data = StringIO(csv_obj)
            df = pd.read_csv(data, delimiter=',')
            df_all = pd.concat([df, df_all], ignore_index=True)
        
        return df_all
    
    #Se crea el método para obtener los objetos que cumplen con la condición de la fecha
    def return_objects(self):
        objects = [obj for obj in self.__bucket.objects.all() if datetime.strptime(obj.key.split("/")[0], '%Y-%m-%d').date() >= self.date]
        return objects
    
    #Se crea el método para ejecutar las funciones de la clase
    def run(self):
        print('Extrayendo la información...')
        objects = self.return_objects()
        df_all = self.read_csv_to_df(objects)
        return df_all
    

class Transform():
    def __init__(self, dataframe):
        self.dataframe = dataframe
        
    #Se crea el método para transformar la información entrante
    def run(self):
        print('Transformando la información...')
        self.dataframe.dropna(inplace=True)
        self.dataframe['start_price'] = self.dataframe.sort_values(by=['Time']).groupby(['ISIN', 'Date'])['StartPrice'].transform('first')
        self.dataframe['end_price'] = self.dataframe.sort_values(by=['Time']).groupby(['ISIN', 'Date'])['EndPrice'].transform('last')
        df_all = self.dataframe.query('"08:00" < Time < "12:00"').groupby(['ISIN', 'Date'], as_index=False).agg(start_price=('start_price', 'min'), end_price=('end_price', 'min'), minimum_price=('MinPrice', 'min'), maximum_price=('MaxPrice', 'max'), daily_traded_volume=('TradedVolume', 'sum'))
        df_all["end_price_mx"] = df_all["end_price"] * 19.08
        deviation = ['start_price','end_price']
        df_all["standard_deviation"] = df_all[deviation].std(axis=1)
        return df_all
    

class Load():
    def __init__(self, dataframe, target_bucket_name):
        self.dataframe = dataframe
        self.__s3 = boto3.resource('s3')
        self.bucket_target = self.__s3.Bucket(target_bucket_name)

    #Se crea el método para escribir el dataframe en un archivo parquet y subirlo a S3
    def write_df_to_s3(self, key):
        out_buffer = BytesIO()
        self.dataframe.to_parquet(out_buffer, index=False)
        self.bucket_target.put_object(Body=out_buffer.getvalue(), Key=key)

    #Se crea el método para ejecutar las funciones de la clase    
    def run(self):
        print('Subiendo la información...')
        key = 'xetra_daily_report_' + datetime.today().strftime("%Y%m%d_%H%M%S") + '.parquet'
        self.write_df_to_s3(key)
        print('La información se subió correctamente')
        return key
    

class XetraReporte(Load):
    def __init__(self, loaded_parquet_key, target_bucket_name):
        self.__s3 = boto3.resource('s3')
        self.bucket_target = self.__s3.Bucket(target_bucket_name)
        self.loaded_parquet_key = loaded_parquet_key

    #Se crea el método para leer el archivo parquet y convertirlo en un dataframe
    def etl_report(self):
        print('Generando el reporte...')
        prq_obj = self.bucket_target.Object(key=self.loaded_parquet_key).get().get('Body').read()
        data = BytesIO(prq_obj)
        df_report = pd.read_parquet(data)
        print('El reporte del parquet: '+self.loaded_parquet_key+', se generó correctamente.')
        print('----------------------------------------------------------')
        return df_report
    

# Crear una instancia de la clase Extract
clase_extract = Extract(bucket_name='xetra-1234', date='2022-12-31')
# Crear una instancia de la clase Transform
clase_transform = Transform(clase_extract.run())
# Crear una instancia de la clase Load
clase_load = Load(clase_transform.run(), target_bucket_name='xetra-ajlj')
# Crear una instancia de la clase XetraReporte
clase_reporte = XetraReporte(loaded_parquet_key=clase_load.run(), target_bucket_name='xetra-ajlj')

print(clase_reporte.etl_report())