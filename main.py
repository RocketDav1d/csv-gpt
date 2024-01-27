import boto3
import asyncio
import os
import pandas as pd
from dotenv import load_dotenv
import sqlite3
from io import BytesIO
from langchain_openai import OpenAI
from langchain_community.utilities import SQLDatabase
from langchain_experimental.sql import SQLDatabaseChain
load_dotenv()

s3_access_key_id = os.environ.get("S3_ACCESS_KEY_ID")
s3_access_key_secret = os.environ.get("S3_ACCESS_KEY_SECRET")
s3_bucket_name = os.environ.get("S3_BUCKET_NAME")
file_key="NetflixTVShowsAndMovies.csv"


def retrieve_file_from_s3(file_key):
    print(file_key)
    if file_key.endswith('csv'): 
        file_format = 'csv'
    elif file_key.endswith('xlsx'):
        file_format = 'xlsx'
    else:
        raise ValueError("Unsupported file format")
    # Connect to the S3 bucket
    s3_client = boto3.client(
            's3',
            aws_access_key_id=s3_access_key_id,
            aws_secret_access_key=s3_access_key_secret
        )

    response = s3_client.get_object(Bucket=s3_bucket_name, Key=file_key)
    s3_data = response['Body'].read()

    print("file_format")
    print(file_format)

    return s3_data, file_format, file_key
 

def database_exists(db_name):
    """Check if the SQLite database file already exists."""
    return os.path.isfile(db_name)



def create_database_from_s3_data(s3_data, file_format, file_key):
    # Read the data into a DataFrame
    if file_format == 'csv':
        df = pd.read_csv(BytesIO(s3_data))
    elif file_format == 'xlsx':
        df = pd.read_excel(BytesIO(s3_data))
    else:
        raise ValueError("Unsupported file format")

    db_name = file_key.split('.')[0] + '.db'
    # Create a connection to the SQLite database
        # Check if the database already exists
    if not database_exists(db_name):
        # Create a connection to the SQLite database
        conn = sqlite3.connect(db_name)

        # Get the table name from the file key
        table_name = file_key.split('.')[0]
        # Transfer the DataFrame to the SQLite database
        df.to_sql(table_name, conn, if_exists='replace', index=False)

        # Close the connection
        conn.close()
    else:
        print(f"Database '{db_name}' already exists. No new database created.")
    # conn = sqlite3.connect(db_name)
    # table_name = file_key.split('.')[0]
    # df.to_sql(table_name, conn, if_exists='replace', index=False)
    # conn.close()


def read_sql_query(sql, db):
    conn = sqlite3.connect(db)
    cur = conn.cursor()
    cur.execute(sql)
    rows = cur.fetchall()
    for row in rows:
        print(row)
    conn.close()
     

def create_db_agent():
    input_db = SQLDatabase.from_uri('sqlite:///fashion_db.sqlite')
    llm_1 = OpenAI(temperature=0)
    db_agent = SQLDatabaseChain(llm = llm_1,
                            database = input_db,
                            verbose=True)
    return db_agent


def run_db_agent(db_agent, query):
    result = db_agent.run(query)
    print(result)
    return result






# s3_data, file_format, file_key = retrieve_file_from_s3(file_key)
# create_database_from_s3_data(s3_data, file_format, file_key)


# query_input = input("Enter your query: ")
# db_agent = create_db_agent()
# run_db_agent(db_agent, query_input)



