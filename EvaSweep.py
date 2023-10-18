import re
import pandas as pd
import evadb
import os
import openai
import tokenize
import io
import numpy as np
import warnings
from evadb.configuration.constants import EvaDB_INSTALLATION_DIR
import requests
import base64
import time

start_time = time.time()

warnings.filterwarnings("ignore")
pd.set_option('display.max_columns', None)
pd.set_option('display.expand_frame_repr', False)
pd.set_option('max_colwidth', None)

os.environ["OPENAI_KEY"] = "sk---"
openai.api_key = os.environ["OPENAI_KEY"]


# Connect to EvaDB and create table
cursor = evadb.connect().cursor()


#Create the necessary tables
cursor.query("DROP FUNCTION IF EXISTS STR2ARRAY;").df()
cursor.query(f"""
    CREATE FUNCTION STR2ARRAY
    IMPL '{EvaDB_INSTALLATION_DIR}/functions/STR2ARRAY.py'
""").df()

cursor.query("""
  DROP TABLE IF EXISTS code_embeddings_table
""").df()

cursor.query("""
    CREATE TABLE code_embeddings_table (code_snippet TEXT(100), embedding NDARRAY FLOAT32(1,1536))
""").df()

cursor.query("""
  DROP TABLE IF EXISTS inputs
""").df()

cursor.query("""
    CREATE TABLE inputs (input TEXT(500), embed NDARRAY FLOAT32(1,1536))
""").df()

cursor.query("""
  DROP TABLE IF EXISTS response
""").df()

cursor.query("""
    CREATE TABLE response (
    response TEXT(200))
                """).df()

# Tokenize the code
def get_tokens(code):
    tokens = []
    for tok in tokenize.tokenize(io.BytesIO(code.encode('utf-8')).readline):
        tokens.append(tok.string)
    return ' '.join(tokens)

# Preprocess Python code
def preprocess_python_code(code):
    # Remove comments
    code = re.sub(r'#.*?\n', '\n', code)
    code = re.sub(r"'''(.*?)'''", '', code, flags=re.DOTALL)  
    code = re.sub(r'"""(.*?)"""', '', code, flags=re.DOTALL) 

    # Remove single quotes 
    code = code.replace("'", "")
    return code.strip()

# Generate embeddings with OpenAI
def get_code_embedding(code: str) -> list:
    tokenized_code = get_tokens(code)
    response = openai.Embedding.create(input=tokenized_code, model="text-embedding-ada-002")
    embedding = response['data'][0]['embedding']
    embedding = np.array(embedding).reshape(1,-1)
    return embedding

# Insert code embedding into the database
def insert_code_embedding(code_snippet):
    embedding = get_code_embedding(code_snippet).tolist()
    cursor.query(f"""
        INSERT INTO code_embeddings_table (code_snippet, embedding)
        VALUES ('{code_snippet}', '{embedding}');
    """).df()


BASE_URL = "https://api.github.com"
def get_file_content(owner, repo, path,):
    url = f"{BASE_URL}/repos/{owner}/{repo}/contents/{path}"
    response = requests.get(url)
    content = response.json().get('content')
    
    if content:
        return base64.b64decode(content).decode('utf-8')
    return None

def list_files_in_repo(owner, repo, path=""):
    url = f"{BASE_URL}/repos/{owner}/{repo}/contents/{path}"
    response = requests.get(url)
    contents = response.json()
    
    files = []
    
    for item in contents:
        if item['type'] == 'file':
            file_content = get_file_content(owner, repo, item['path'])
            files.append({
                "path": item['path'],
                "content": file_content
            })
        elif item['type'] == 'dir':
            files.extend(list_files_in_repo(owner, repo, item['path']))
    
    return files

# Example Usage
owner = ""
repo = ""
all_files = list_files_in_repo(owner, repo)
count = 0
for file_info in all_files:
    code_snippet = preprocess_python_code(file_info['content'])
    insert_code_embedding(code_snippet)


#After files are created, create the vector index
result = cursor.query
cursor.query("""
    CREATE INDEX IF NOT EXISTS code_embedding_index
    ON code_embeddings_table (STR2ARRAY(embedding))
    USING FAISS;
""").df()

query = f"""
    SELECT code_snippet FROM code_embeddings_table ORDER BY
    Similarity(
      STR2ARRAY(embedding),
      STR2ARRAY(embedding)
    ) DESC
    LIMIT 2
"""
code_snippets = cursor.query(query).df()

cursor.query(f"""INSERT INTO response(response) VALUES ('{code_snippets}');""").df()

query = cursor.query("""SELECT ChatGPT("Provide the corrected code for the following snippet, find and fix bugs,
 without any explanations or additional text:" ,response) FROM response;""").df()
response = query.at[0, 'chatgpt.response']

#write response to a file
with open('output.txt', 'w') as f:
    f.write(response)
