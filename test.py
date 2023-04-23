import boto3
import pandas as pd
from io import StringIO, BytesIO
from datetime import datetime, timedelta


# Adapter Layer

def read_csv_to_df(bucket, objects):

    # Obtener el objeto por posición en el bucket objects
    csv_obj_init = bucket.Object(key=objects[34].key).get().get('Body').read().decode('utf-8')
    data = StringIO(csv_obj_init)
    df_init = pd.read_csv(data, delimiter=',')

    # Se inicializa el nombre de las columnas extrayendo los nombres de df_init
    df_all = pd.DataFrame(columns=df_init.columns)
    return df_all

def write_df_to_s3():
    
    pass

def return_objects():
    pass