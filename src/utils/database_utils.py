class db_connection:
    connection = None

    @staticmethod
    def get_connection():
        connection = pyodbc.connect(
            "DRIVER={ODBC Driver 17 for SQL Server};SERVER=" # 
            + db_configer.get("database", "server")
            + ";DATABASE="
            + db_configer.get("database", "database")
            + ";ENCRYPT=yes;UID="
            + db_configer.get("database", "username")
            + ";PWD="
            + db_configer.get("database", "password")
            + ";TrustServerCertificate=yes;"
        )
        return connection

    def test_connection(self):
        try:
            conn = self.connection()
            print(f"connection is working")
            conn.close()
        except pyodbc.Error as ex:
            print(f'Connection not working\n{ex}')

    def query_data(self, sql_query):
        conn = self.connection()

        cursor = conn.cursor()
        cursor.execute(sql_query)

        # Fetch all the rows that match the conditions
        data = cursor.fetchall()
        columns = [column[0] for column in cursor.description]

        # Close the cursor and the connection
        cursor.close()
        conn.close()
        return data, columns