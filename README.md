# LBRCE College Chatbot

This is a FastAPI-based chatbot designed to provide quick and professional responses to queries related to LBRCE College. It uses LlamaIndex for vector-based search and Meta Llama 3-8B as its underlying model.

## Features
- Chatbot trained on LBRCE College's data
- Context-aware query handling
- Secure and optimized FastAPI backend
- Chat history storage and summarization
- Webhook integration for form submissions
- CORS support for cross-origin requests

## Technologies Used
- **FastAPI**: For building the web API
- **Hugging Face Inference API**: For LLM processing
- **LlamaIndex**: For document ingestion and querying
- **Jinja2**: For template rendering
- **Gradio Client**: For interacting with AI models
- **Docker**: For containerized deployment

## Setup Instructions

### Prerequisites
- Python 3.12+
- A Hugging Face API token (stored as an environment variable `HF_TOKEN`)
- Docker (optional, for containerized deployment)

### Installation
1. Clone the repository:
   ```sh
   git clone https://github.com/Srinivasulu2003/Lbrce-College-Chatbot.git
   cd Lbrce-College-Chatbot
   ```
2. Create and activate a virtual environment:
   ```sh
   python -m venv venv
   source venv/bin/activate  # On Windows use `venv\Scripts\activate`
   ```
3. Install dependencies:
   ```sh
   pip install -r requirements.txt
   ```

### Running the Application
1. Set up environment variables:
   ```sh
   export HF_TOKEN=your_huggingface_token
   ```
2. Start the FastAPI server:
   ```sh
   uvicorn app:app --host 0.0.0.0 --port 7860
   ```
3. Access the chatbot at `http://localhost:7860`

### Running with Docker
1. Build the Docker image:
   ```sh
   docker build -t srinu-college-chatbot .
   ```
2. Run the container:
   ```sh
   docker run -p 7860:7860 srinu-college-chatbot
   ```

## API Endpoints
| Method | Endpoint         | Description                        |
|--------|----------------|------------------------------------|
| GET    | `/`            | Loads the homepage                 |
| POST   | `/chat/`       | Handles user chat queries         |
| GET    | `/ch/{id}`     | Loads a chat session              |
| POST   | `/hist/`       | Saves chat history                |
| POST   | `/webhook`     | Receives and processes form data  |

## Folder Structure
```
/app
│── static/          # Static files
│── data/            # Data storage directory
│── db/              # Vector index storage
│── app.py           # Main application script
│── requirements.txt # Dependencies
│── Dockerfile       # Docker setup
│── README.md        # Project documentation
```

## License
This project is licensed under the MIT License.

## Contributors
- **SRINIVASULU KETHANABOINA** - Developer & Maintainer

For queries and contributions, feel free to open an issue or pull request.

