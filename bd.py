import psycopg2
from psycopg2 import Error

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