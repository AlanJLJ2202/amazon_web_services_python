import boto3
import pandas as pd
import numpy as np
from io import StringIO, BytesIO
from datetime import datetime, timedelta
from sklearn.linear_model import LinearRegression


class ETL():
    def __init__(self, bucket_name, target_bucket_name, date):
        self.__s3 = boto3.resource('s3')
        self.__bucket = self.__s3.Bucket(bucket_name)
        self.bucket_target = self.__s3.Bucket(target_bucket_name)
        self.date = datetime.strptime(date, '%Y-%m-%d').date() - timedelta(days=1)
    
    
    def extract(self):
        pass
    
    def transform(self, df_all):
        pass
    
    def load(self, df_all):
        pass
    
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
    
    def write_df_to_s3(self, df_all, key):
        out_buffer = BytesIO()
        df_all.to_parquet(out_buffer, index=False)
        self.bucket_target.put_object(Body=out_buffer.getvalue(), Key=key)
    
    def return_objects(self):
        objects = [obj for obj in self.__bucket.objects.all() if datetime.strptime(obj.key.split("/")[0], '%Y-%m-%d').date() >= self.date]
        return objects
    

class XetraReporte(ETL):
    def __init__(self, bucket_name, target_bucket_name, date):
        super().__init__(bucket_name, target_bucket_name, date)
    
    def extract(self):
        objects = self.return_objects()
        df_all = self.read_csv_to_df(objects)
        return df_all
    
    def transform(self, df_all):
        df_all.dropna(inplace=True)
        df_all['start_price'] = df_all.sort_values(by=['Time']).groupby(['ISIN', 'Date'])['StartPrice'].transform('first')
        df_all['end_price'] = df_all.sort_values(by=['Time']).groupby(['ISIN', 'Date'])['EndPrice'].transform('last')
        df_all = df_all.query('"08:00" < Time < "12:00"').groupby(['ISIN', 'Date'], as_index=False).agg(start_price=('start_price', 'min'), end_price=('end_price', 'min'), minimum_price=('MinPrice', 'min'), maximum_price=('MaxPrice', 'max'), daily_traded_volume=('TradedVolume', 'sum'))
        df_all["end_price_mx"] = df_all["end_price"] * 19.08
        deviation = ['start_price','end_price']
        df_all["standard_deviation"] = df_all[deviation].std(axis=1)
        return df_all
    

    def load(self, df_all):
        key = 'xetra_daily_report_' + datetime.today().strftime("%Y%m%d_%H%M%S") + '.parquet'
        self.write_df_to_s3(df_all, key)

    def etl_report(self, key):
        prq_obj = self.bucket_target.Object(key=key).get().get('Body').read()
        data = BytesIO(prq_obj)
        df_report = pd.read_parquet(data)
        return df_report

    def run(self):
        df_all = self.extract()
        df_transformed = self.transform(df_all)
        model = self.linear_regression(df_transformed)
        print(model)
        print('DATAFRAME TRANSFORMADO:')
        print(df_transformed)
        self.load(df_transformed)


    def linear_regression(self, df_all):
        # Selecciona la columna 'end_price' donde la fecha se encuentre entre 'arg_date' y 'arg_date + 1 día'
        y = df_all.query(f'Date == "{self.date}"')['end_price']
        
        # Crea una matriz X con los valores de la columna 'start_price'
        x = np.array(df_all.query(f'Date == "{self.date}"')['start_price']).reshape(-1, 1)
        
        # Crea una instancia de la clase LinearRegression
        model = LinearRegression()
        
        # Entrena el modelo con los datos
        model.fit(x, y)
        
        print('COEFICIENTES:')
        print(model.coef_)
        print('INTERCEPTO DE MODELO:')
        print(model.intercept_)
        
        return model
    

# Crear una instancia de la clase
report = XetraReporte(bucket_name='xetra-1234', target_bucket_name='xetra-ajlj', date='2022-12-31')

# Ejecutar el proceso ETL
report.run()

# Obtener el reporte
key = 'xetra_daily_report_20230328_000509.parquet'
df_report = report.etl_report(key)
