from fundamental.lib.connectors.dbconnector import DBconnector

class CliClient:
    def __init__(self):
        self.db_connector = DBconnector()

    def query(self, query):
        return self.db_connector.query(query=query)


if __name__ == '__main__':
    cli_client = CliClient()
    query = f"""
    DROP TABLE IF EXISTS signals;
    """
    cli_client.db_connector.perform_query(query=query)
    query = f"""
    CREATE TABLE signals (
        id SERIAL PRIMARY KEY,
        ticker VARCHAR(10),
        price FLOAT,
        signal_type VARCHAR(100),
        strength INT,
        date DATE,
        description VARCHAR(100)
    );
    """
    print(query)
    recs = cli_client.db_connector.perform_query(query=query)
    print(recs)