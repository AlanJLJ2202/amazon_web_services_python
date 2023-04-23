import boto3
import pandas as pd
import numpy as np
from io import StringIO, BytesIO
from datetime import datetime, timedelta
from sklearn.linear_model import LinearRegression

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
    
class ApplicationLayer(ETL):
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

al = ApplicationLayer(arg_date='2022-12-31', bucket_name='xetra-1234', bucket_target_name='xetra-cdhm')
#df = al.extract()
#transformed_data = al.transform_report(df_all=df)

#al.load_report(df_all=transformed_data)
report = al.etl_report()
print(report)


