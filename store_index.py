from src.helper import repo_ingestion, load_repo, text_splitter, load_embedding
from dotenv import load_dotenv
from langchain_community.vectorstores import Chroma
import os
import uuid

load_dotenv()

OPENAI_API_KEY = os.environ.get('OPENAI_API_KEY')
os.environ["OPENAI_API_KEY"] = OPENAI_API_KEY


documents = load_repo("repo/")
text_chunks = text_splitter(documents)
embeddings = load_embedding()


# storing vector in choramdb
#ids = [i for i in range(0, len(text_chunks), 1)]   # Generate unique IDs before creating the database
#ids = ['1', '1', '1', '1', '1', '1' , '1', '1', '1', '1', '1', '1', '1', '1']
ids = [print(i) for i in range(1, len(text_chunks) + 1)]
print(ids)
vectordb = Chroma.from_documents(text_chunks, embedding=embeddings, ids=ids, persist_directory='./db')
vectordb.persist()  # Persist after creating the database

