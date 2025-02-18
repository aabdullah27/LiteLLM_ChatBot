# Document Chat Comparison

## Overview
Document Chat Comparison is a Streamlit application that allows users to upload documents (PDF or text files) and ask questions about them while comparing responses from two leading AI models: Groq and Gemini. This tool is particularly useful for researchers, content creators, and anyone interested in comparing how different AI models interpret and respond to queries about document content.

## Features
- **Document Processing**: Upload PDF or text files and process them for semantic search
- **Dual Model Comparison**: Compare responses from Groq and Gemini models side-by-side
- **Flexible Model Selection**: Choose from various model versions for both Groq and Gemini
- **Conversational History**: Track conversation history with both models
- **Efficient Processing**: Document processing with caching to avoid redundant operations
- **Asynchronous Operations**: Uses async processing for better performance

## Installation

### Prerequisites
- Python 3.8 or later
- API keys for both Groq and Gemini services

### Setup
1. Clone this repository or download the source code
2. Create a virtual environment (recommended):
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```
3. Install required packages:
   ```bash
   pip install -r requirements.txt
   ```

### Required Libraries
The application depends on the following libraries:
- streamlit: Web application framework
- langchain, langchain-community, langchain-core: Framework for building applications with language models
- langchain-google-genai: Integration with Google's Generative AI models
- faiss-cpu: Vector database for efficient similarity search
- google-generativeai: Google's Generative AI API
- litellm: Library for accessing various LLM APIs
- pymupdf, pymupdf4llm: Libraries for PDF processing
- python-dotenv: Environment variable management
- asyncio: Asynchronous I/O library
- groq: Groq API client

## Usage

### Starting the Application
```bash
streamlit run app.py
```

### Configuration
1. Enter your API keys in the sidebar:
   - Groq API Key
   - Gemini API Key
2. Upload a document (PDF or TXT format)
3. Select the models you want to compare:
   - Groq models: deepseek-r1-distill-llama-70b, llama-3.3-70b-versatile, mixtral-8x7b-32768, llama3-8b-8192
   - Gemini models: gemini-2.0-flash-exp, gemini-1.5-flash-8b, gemini-1.5-pro

### Asking Questions
1. Wait for the document to be processed (you'll see a success message)
2. Type your question in the chat input field
3. View the side-by-side responses from both models

### Clearing History
Use the "Clear Chat History" button in the sidebar to reset the conversation.

## How It Works

### Document Processing
1. The application uploads and processes the document asynchronously
2. For PDFs, the content is converted to markdown using pymupdf4llm
3. For text files, the content is processed directly
4. The text is split into chunks using RecursiveCharacterTextSplitter
5. Embeddings are created using Google's embedding model
6. A FAISS vector store is created for efficient retrieval

### Question Answering
1. When a question is asked, the application retrieves relevant document chunks
2. A prompt template is filled with the context and question
3. Both models (Groq and Gemini) process the prompt concurrently
4. Responses are displayed side-by-side for comparison

## Architecture
- **Frontend**: Streamlit provides the web interface
- **Document Processing**: Uses langchain and pymupdf for processing documents
- **Embeddings**: Google's embedding model creates vector representations
- **Vector Store**: FAISS enables efficient semantic search
- **Model Integration**: Groq and Gemini APIs are accessed through their respective clients
- **Asynchronous Operations**: asyncio enables concurrent processing and API calls

## Performance Considerations
- Document processing is cached to avoid reprocessing the same document
- Text splitter is cached using lru_cache for better performance
- Asynchronous calls to both models happen concurrently
