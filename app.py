import os
import time
from fastapi import FastAPI
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from llama_index.core import StorageContext, load_index_from_storage, VectorStoreIndex, SimpleDirectoryReader, ChatPromptTemplate, Settings
from llama_index.llms.huggingface import HuggingFaceInferenceAPI
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from pydantic import BaseModel
import datetime
from dotenv import load_dotenv
load_dotenv()
# Define Pydantic model for incoming request body
class MessageRequest(BaseModel):
    message: str


os.environ["HF_TOKEN"] = os.getenv("HF_TOKEN")
app = FastAPI()

app.mount("/static", StaticFiles(directory="static"), name="static")


# Configure Llama index settings
Settings.llm = HuggingFaceInferenceAPI(
    model_name="meta-llama/Meta-Llama-3-8B-Instruct",
    tokenizer_name="meta-llama/Meta-Llama-3-8B-Instruct",
    context_window=3000,
    token=os.getenv("HF_TOKEN"),
    max_new_tokens=512,
    generate_kwargs={"temperature": 0.1},
)
Settings.embed_model = HuggingFaceEmbedding(
    model_name="BAAI/bge-small-en-v1.5"
)

PERSIST_DIR = "db"
PDF_DIRECTORY = 'data'

# Ensure directories exist
os.makedirs(PDF_DIRECTORY, exist_ok=True)
os.makedirs(PERSIST_DIR, exist_ok=True)
chat_history = []
current_chat_history = []
def data_ingestion_from_directory():
    documents = SimpleDirectoryReader(PDF_DIRECTORY).load_data()
    storage_context = StorageContext.from_defaults()
    index = VectorStoreIndex.from_documents(documents)
    index.storage_context.persist(persist_dir=PERSIST_DIR)

def initialize():
    start_time = time.time()
    data_ingestion_from_directory()  # Process PDF ingestion at startup
    print(f"Data ingestion time: {time.time() - start_time} seconds")

initialize()  # Run initialization tasks


def handle_query(query):
    chat_text_qa_msgs = [
        (
            "user",
            """
            You are the Clara Redfernstech chatbot. Your goal is to provide accurate, professional, and helpful answers to user queries based on the company's data. Always ensure your responses are clear and concise. Give response within 10-15 words only       
            {context_str}
            Question:
            {query_str}
            """
        )
    ]
    text_qa_template = ChatPromptTemplate.from_messages(chat_text_qa_msgs)
    
    storage_context = StorageContext.from_defaults(persist_dir=PERSIST_DIR)
    index = load_index_from_storage(storage_context)
    context_str = ""
    for past_query, response in reversed(current_chat_history):
        if past_query.strip():
            context_str += f"User asked: '{past_query}'\nBot answered: '{response}'\n"

    
    query_engine = index.as_query_engine(text_qa_template=text_qa_template, context_str=context_str)
    answer = query_engine.query(query)

    if hasattr(answer, 'response'):
        response=answer.response
    elif isinstance(answer, dict) and 'response' in answer:
        response =answer['response']
    else:
        response ="Sorry, I couldn't find an answer."
    current_chat_history.append((query, response))
    return response
@app.get("/", response_class=HTMLResponse)
async def read_root():
    with open("static/index.html") as f:
        return f.read()

@app.post("/chat/")
async def chat(request: MessageRequest):
    message = request.message  # Access the message from the request body
    response = handle_query(message)  # Process the message
    message_data = {
        "sender": "User",
        "message": message,
        "response": response,
        "timestamp": datetime.datetime.now().isoformat()
    }
    chat_history.append(message_data)
    return {"response": response}


