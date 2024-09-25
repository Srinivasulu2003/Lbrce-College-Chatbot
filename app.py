import os
import time
from fastapi import FastAPI,Request
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from llama_index.core import StorageContext, load_index_from_storage, VectorStoreIndex, SimpleDirectoryReader, ChatPromptTemplate, Settings
from llama_index.llms.huggingface import HuggingFaceInferenceAPI
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from pydantic import BaseModel
from fastapi.responses import JSONResponse
import uuid  # for generating unique IDs
import datetime
from fastapi.middleware.cors import CORSMiddleware
from fastapi.templating import Jinja2Templates
from huggingface_hub import InferenceClient
import json
import re
from gradio_client import Client
from simple_salesforce import Salesforce, SalesforceLogin


# Define Pydantic model for incoming request body
class MessageRequest(BaseModel):
    message: str
repo_id = "meta-llama/Meta-Llama-3-8B-Instruct"
llm_client = InferenceClient(
    model=repo_id,
    token=os.getenv("HF_TOKEN"),
)
client = Client("Be-Bo/llama-3-chatbot_70b")
os.environ["HF_TOKEN"] = os.getenv("HF_TOKEN")
username = os.getenv("username")
password = os.getenv("password")
security_token = os.getenv("security_token")
domain =  os.getenv("domain")# Using sandbox environment


app = FastAPI()


@app.middleware("http")
async def add_security_headers(request: Request, call_next):
    response = await call_next(request)
    response.headers["Content-Security-Policy"] = "frame-ancestors *; frame-src *; object-src *;"
    response.headers["X-Frame-Options"] = "ALLOWALL"
    return response


# Allow CORS requests from any domain
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)




@app.get("/favicon.ico")
async def favicon():
    return HTMLResponse("")  # or serve a real favicon if you have one


app.mount("/static", StaticFiles(directory="static"), name="static")

templates = Jinja2Templates(directory="static")
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
def split_name(full_name):
    # Split the name by spaces
    words = full_name.strip().split()
    
    # Logic for determining first name and last name
    if len(words) == 1:
        first_name = ''
        last_name = words[0]
    elif len(words) == 2:
        first_name = words[0]
        last_name = words[1]
    else:
        first_name = words[0]
        last_name = ' '.join(words[1:])
    
    return first_name, last_name

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
@app.get("/ch/{id}", response_class=HTMLResponse)
async def load_chat(request: Request, id: str):
    return templates.TemplateResponse("index.html", {"request": request, "user_id": id})
# Route to save chat history
@app.post("/hist/")
async def save_chat_history(history: dict):
    # Check if 'userId' is present in the incoming dictionary
    user_id = history.get('userId')
    print(user_id)

    # Ensure user_id is defined before proceeding
    if user_id is None:
        return {"error": "userId is required"}, 400

    # Construct the chat history string
    hist = ''.join([f"'{entry['sender']}: {entry['message']}'\n" for entry in history['history']])
    hist = "You are a Redfernstech summarize model. Your aim is to use this conversation to identify user interests solely based on that conversation: " + hist
    print(hist)

    # Get the summarized result from the client model
    result = client.predict(
        message=hist,
        api_name="/chat"
    )

    try:
        sf.Lead.update(user_id, {'Description': result})
    except Exception as e:
        return {"error": f"Failed to update lead: {str(e)}"}, 500
    
    return {"summary": result, "message": "Chat history saved"}
@app.post("/webhook")
async def receive_form_data(request: Request):
    form_data = await request.json()
    # Log in to Salesforce
    session_id, sf_instance = SalesforceLogin(username=username, password=password, security_token=security_token, domain=domain)

    # Create Salesforce object
    sf = Salesforce(instance=sf_instance, session_id=session_id)
    first_name, last_name = split_name(form_data['name'])
    data = {
    'FirstName': first_name,
    'LastName': last_name,
    'Description': 'hii',  # Static description
    'Company': form_data['company'],  # Assuming company is available in form_data
    'Phone': form_data['phone'].strip(),  # Phone from form data
    'Email': form_data['email'],  # Email from form data
    }
    a=sf.Lead.create(data)
    # Generate a unique ID (for tracking user)
    unique_id = a['id']
    
    # Here you can do something with form_data like saving it to a database
    print("Received form data:", form_data)
    
    # Send back the unique id to the frontend
    return JSONResponse({"id": unique_id})

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
@app.get("/")
def read_root():
    return {"message": "Welcome to the API"}
