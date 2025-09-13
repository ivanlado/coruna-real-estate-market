import psycopg2
from psycopg2 import Error
import pandas as pd

cols_1 = ['id', 'titulo', 'descripcion', 'extra_info', 'n_habitaciones', 'tamano', 'cast(precio as numeric) as precio', 'municipio', 'n_banos', 'n_plazas_garaje', 'direccion', 'landmarks_cercanos', 'piscina', 'valoracion', 'direccion_x', 'direccion_y']
cols_2 = ['id', 'direccion', 'municipio']

class SQLDatabase:
    def __init__(self, user, password, host, port, db_name, auto_commit=True):
        self.user = user
        self.password = password
        self.host = host
        self.port = port
        self.db_name = db_name
        self.connection = None
        self.cursor = None
        print("Database object created")

    def connect(self):
        if self.connection == None: 
            try:
                self.connection = psycopg2.connect(user=self.user, password=self.password, host=self.host, port=self.port, database=self.db_name)
                self.connection.autocommit = True
                self.cursor = self.connection.cursor()
            except (Exception, Error) as error:
                print("Error while connecting to PostgreSQL", error)

    def disconnect(self):
        if self.connection:
            self.connection.close()
        else:
            print("No active connection to close.")

    def get_connection(self):
        if self.connection == None:
            self.connect()
        return self.connection
    
    def get_cursor(self):
        return self.cursor
    
    def get_data_v1(self, cols = cols_1):
        # Esto no es seguro, no previene SQL injection
        col_list = ", ".join(cols) 
        query = f'SELECT {col_list} FROM (viviendas LEFT JOIN viviendas_info_gpt USING (id)) LEFT join viviendas_direcciones USING (id)'
        df = pd.read_sql_query(query, self.connection)
        return df
    
    def insert_coordenadas(self, id, x, y, es_exacta):
        query = "INSERT INTO public.viviendas_direcciones(id, direccion_x, direccion_y, es_exacta) VALUES (%s, %s, %s, %s);"
        try:
            self.cursor.execute(query, (id, x, y, es_exacta))
        except (Exception, Error) as error:
            return False
        return True
    
    def get_ids_sin_direccion(self):
        query = "SELECT id FROM viviendas WHERE id NOT IN (SELECT id FROM viviendas_direcciones);"
        df = pd.read_sql_query(query, self.connection)
        return df['id'].values.tolist()     