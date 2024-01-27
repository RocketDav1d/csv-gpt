import streamlit as st
import pandas as pd
import numpy as np
from streamlit_option_menu import option_menu
import boto3
from io import BytesIO
import re
import sqlite3
import os
import io
from dotenv import load_dotenv
import openai
from langchain_openai import OpenAI, ChatOpenAI
from langchain.agents.agent_types import AgentType
from langchain_community.utilities import SQLDatabase
from langchain_experimental.sql import SQLDatabaseChain
from langchain.chains.sql_database.query import create_sql_query_chain
from langchain_experimental.agents.agent_toolkits import create_csv_agent
from langchain_experimental.agents.agent_toolkits import create_pandas_dataframe_agent


function_descriptions = [
    {
        "name": "create_dataframe_response",
        "description": "Create a DataFrame response by executing an SQL query generated from the provided query.",
        "parameters": {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "The user-provided query to generate an SQL query."
                },
                "file_key": {
                    "type": "object",
                    "description": "The S3 bucket key of the file used to retrieve the data and create a sqlite database."
                }, 
            },
            "required": ["query", "file_key"],
        },
    },
    {
        "name": "create_text_response",
        "description": "Create a text response by running a query through the database agent.",
        "parameters": {
            "type": "object",
            "properties": {
                 "query": {
                    "type": "string",
                    "description": "The query to be processed by the database agent.",
                },
                "df": {
                    "type": "object",
                    "description": "The dataframe to be processed by the dataframe agent.",
                },

            },
            "required": ["query", "df"],
        },
    },
]


load_dotenv()
s3_access_key_id = os.environ.get("S3_ACCESS_KEY_ID")
s3_access_key_secret = os.environ.get("S3_ACCESS_KEY_SECRET")
file_key="NetflixTVShowsAndMovies.csv"
s3_bucket_name = "sheets-gpt-project"


st.set_page_config(layout="wide")
if "visibility" not in st.session_state:
    st.session_state.visibility = "visible"
    st.session_state.disabled = False
    st.session_state.placeholder = "e.g. 'How many movies have a runtime below 90 minutes?'"
col1, col2 = st.columns(2)

style = """
        <style>
        .highlight {
            border: 2px solid black;
            border-radius: 10px;
        }
        </style>
        """
st.markdown(style, unsafe_allow_html=True)






s3_client = boto3.client(
        's3',
        aws_access_key_id=s3_access_key_id,
        aws_secret_access_key=s3_access_key_secret
    )

def upload_file_to_s3(bucket, key, file_obj):
    """
    Upload a file-like object to an S3 bucket

    :param bucket: str. Name of the S3 bucket.
    :param key: str. The key under which to store the new file.
    :param file_obj: File-like object to upload.
    """
    try:
        # If the file object is not in bytes, convert it to bytes
        if not isinstance(file_obj, io.BytesIO):
            file_obj = io.BytesIO(file_obj)


        # Use the 'put_object' method to upload the file object
        s3_client.upload_fileobj(file_obj, bucket, key)
        # s3_client.put_object(Bucket=bucket, Key=key, Body=file_obj)
        print("File uploaded successfully")
        st.toast(f"File uploaded successfully to {bucket}/{key}")
        return True, key
    except Exception as e:
        st.error("Failed to upload file to S3: {}".format(e))


def retrieve_file_from_s3(file_key):
    file_format = 'csv' if file_key.endswith('csv') else 'xlsx' if file_key.endswith('xlsx') else None
    if not file_format:
        raise ValueError("Unsupported file format")
    
    response = s3_client.get_object(Bucket=s3_bucket_name, Key=file_key)
    return response['Body'].read(), file_format, file_key


def list_s3_keys(bucket):
    """List keys in an S3 bucket."""
    keys = []
    # Initialize the paginator
    paginator = s3_client.get_paginator('list_objects_v2')
    
    # Create a PageIterator from the Paginator
    page_iterator = paginator.paginate(Bucket=bucket)

    # Loop through each object, appending the keys to a list
    for page in page_iterator:
        if "Contents" in page:
            for obj in page['Contents']:
                keys.append(obj['Key'])
    return keys


def return_csv_agent(csv_file):
    agent = create_csv_agent(
    ChatOpenAI(temperature=0, model="gpt-3.5-turbo-0613"),
    csv_file,
    verbose=True,
    agent_type=AgentType.OPENAI_FUNCTIONS,
    )
    return agent


def return_df_agent(df):
    agent = create_pandas_dataframe_agent(
    ChatOpenAI(temperature=0, model="gpt-3.5-turbo-0613"),
    df,
    verbose=True,
    agent_type=AgentType.OPENAI_FUNCTIONS,
    )
    return agent


# Check if database exists
def database_exists(db_name):
    """Check if the SQLite database file already exists."""
    return os.path.isfile(db_name)

# Create database from S3 data
def create_database_from_s3_data(s3_data, file_format, file_key):
    db_name = file_key.split('.')[0] + '.db'
    if not database_exists(db_name):
        if file_format == 'csv':
            df = pd.read_csv(BytesIO(s3_data))
        elif file_format == 'xlsx':
            df = pd.read_excel(BytesIO(s3_data))
        else:
            raise ValueError("Unsupported file format")
        
        conn = sqlite3.connect(db_name)
        table_name = file_key.split('.')[0]
        df.to_sql(table_name, conn, if_exists='replace', index=False)
        conn.close()
    else:
        print(f"Database '{db_name}' already exists. No new database created.")
    return db_name


def create_dataframe_response(query, file_key):

    s3_data, file_format, file_key = retrieve_file_from_s3(file_key)
    db_name = create_database_from_s3_data(s3_data, file_format, file_key)
    input_db = SQLDatabase.from_uri(f'sqlite:///{db_name}')
    llm_1 = OpenAI(openai_api_key=os.environ.get("OPENAI_API_KEY"), temperature=0)

    sql_query_chain = create_sql_query_chain(llm_1, input_db)
    sql_query = sql_query_chain.invoke({"question": query})
    sql_query = re.sub(r"\sLIMIT\s+\d+", "", sql_query, flags=re.IGNORECASE)
    conn = sqlite3.connect(db_name)
    df = pd.read_sql_query(sql_query, conn)
    conn.close()
    return df

def create_text_response(query, df):
    print("Creating text response")
    # agent = return_csv_agent(csv_file)
    agent = return_df_agent(df)
    result = agent.run(query)
    print(result)
    return result


# Create and run the database agent
def run_db_agent(query, csv_file, df, file_key):
    max_retries = 3  # Set the maximum number of retries
    print("QUERY🟠", query)
    for attempt in range(max_retries):
        completion = openai.chat.completions.create(
        model="gpt-3.5-turbo-0613",
        messages = [
        {"role": "assistant", "content": "You are a helpful assistant. Decide the appropriate response format for each query and call the respective function."},
        {"role": "user", "content": "How many movies have a runtime below 100?"},
        {"role": "system", "content": "{\"function\": \"create_text_response\", \"parameters\": {\"query\": \"SELECT COUNT(*) FROM movies WHERE runtime < 100\", \"df\": \"<df>\"}}"},
        {"role": "user", "content": "What are the movies with a runtime below 100?"},
        {"role": "system", "content": "{\"function\": \"create_dataframe_response\", \"parameters\": {\"query\": \"SELECT * FROM movies WHERE runtime < 100\", \"file_key\": \"<file_key>\"}}"},
        {"role": "user", "content": "Which actors play in the movies with a runtime below 100?"},
        {"role": "system", "content": "{\"function\": \"create_dataframe_response\", \"parameters\": {\"query\": \"SELECT actors FROM movies WHERE runtime < 100\", \"file_key\": \"<file_key>\"}}"},
        {'role': 'user', 'content': query},
        ],
        functions=function_descriptions,
        function_call="auto",  # specify the function call
        )
        output = completion.choices[0].message
        print(output)
        if hasattr(output, 'function_call') and output.function_call:
            query = output.function_call.arguments
            print("Query", query)
            chosen_function = eval(output.function_call.name)
            print("CHOOSEN FUNCTION:", chosen_function)
            if "create_dataframe_response" in str(chosen_function):
                print("✅ CREATE DATAFRAME RESPONSE")
                response = create_dataframe_response(query=query, file_key=file_key)
                st.dataframe(response)
                break
            elif "create_text_response" in str(chosen_function):
                print("✅ CREATE TEXT RESPONSE")
                response = chosen_function(query, df)

                col1, col2 = st.columns([1, 4])
                with col1:
                    st.write("")
                with col2:
                    st.markdown(f'<p class="highlight">{response}</p>', unsafe_allow_html=True)
                break
        else: 
            print("🛑 No function call detected, retrying...")
            if attempt == max_retries - 1:
                st.error("Failed to obtain a function call after several attempts.")


with col1:
    uploaded_files = st.file_uploader("Choose a CSV file", accept_multiple_files=True)
    query = st.text_input(
            "Query",
            label_visibility=st.session_state.visibility,
            disabled=st.session_state.disabled,
            placeholder=st.session_state.placeholder,
        )
    run_query = st.button("Run Query")




for uploaded_file in uploaded_files:
    # Read the file into a DataFrame
    df = pd.read_csv(uploaded_file)
    with col2:
        st.dataframe(df)  # Display the dataframe in the app
    
    if uploaded_files and query and run_query:
        # Upload the file to S3
        uploaded_file.seek(0)
        file_key = uploaded_file.name
        response, file_key = upload_file_to_s3(s3_bucket_name, file_key, uploaded_file)
        if response:
            run_db_agent(query=query, csv_file=uploaded_file, df=df, file_key=file_key)

        # file_obj = uploaded_file.read()  # Read the file-like object
        # file_key = uploaded_file.name    # Use the uploaded file's name as the S3 key
        # upload_file_to_s3(s3_bucket_name, file_key, file_obj)  # Upload the file



with st.sidebar:
    all_keys = list_s3_keys(s3_bucket_name)
    selected = option_menu("Files", all_keys, 
        icons=['house', 'gear'], menu_icon="files", default_index=1)
    selected

