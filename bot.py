# import os
# import streamlit as st
# from langchain.text_splitter import RecursiveCharacterTextSplitter
# from langchain_community.vectorstores import FAISS
# from langchain_google_genai import GoogleGenerativeAIEmbeddings
# from langchain_core.prompts import PromptTemplate
# import tempfile
# from litellm import completion
# import pymupdf4llm
# import asyncio
# from groq import Groq
# from functools import lru_cache

# # Page configuration
# st.set_page_config(page_title="Document Chat Comparison", page_icon="🤖", layout="wide")

# # Initialize session state with a dictionary comprehension
# INITIAL_STATE = {
#     "messages_groq": [],
#     "messages_gemini": [],
#     "vector_store": None,
#     "groq_client": None,
#     "document_hash": None  # Add hash to check if document has changed
# }

# for key, value in INITIAL_STATE.items():
#     if key not in st.session_state:
#         st.session_state[key] = value

# @lru_cache(maxsize=1)
# def get_text_splitter():
#     """Cache the text splitter instance"""
#     return RecursiveCharacterTextSplitter(
#         chunk_size=1000,
#         chunk_overlap=200,
#         separators=["\n\n", "\n", " ", ""]
#     )

# def clear_chat_history():
#     st.session_state.messages_groq = []
#     st.session_state.messages_gemini = []
#     st.rerun()

# async def process_document(file_content, gemini_api_key):
#     """Process document asynchronously"""
#     with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as temp_file:
#         temp_file.write(file_content)
#         temp_file_path = temp_file.name

#     try:
#         # Convert PDF to markdown
#         markdown_text = pymupdf4llm.to_markdown(temp_file_path)
        
#         # Get cached text splitter
#         text_splitter = get_text_splitter()
#         texts = text_splitter.create_documents([markdown_text])
        
#         # Create embeddings instance once
#         embeddings = GoogleGenerativeAIEmbeddings(
#             model="models/embedding-001",
#             google_api_key=gemini_api_key
#         )
        
#         # Create vector store
#         vector_store = FAISS.from_documents(
#             documents=texts,
#             embedding=embeddings
#         )
        
#         return vector_store
    
#     finally:
#         if os.path.exists(temp_file_path):
#             os.remove(temp_file_path)

# async def get_model_responses(formatted_prompt, groq_client, groq_model, gemini_model, gemini_api_key):
#     """Get responses from both models concurrently"""
#     async def get_groq_response():
#         return groq_client.chat.completions.create(
#             model=groq_model,
#             messages=[{"role": "user", "content": formatted_prompt}]
#         )
    
#     async def get_gemini_response():
#         return completion(
#             model=f"gemini/{gemini_model}",
#             messages=[{"role": "user", "content": formatted_prompt}],
#             api_key=gemini_api_key
#         )
    
#     groq_response, gemini_response = await asyncio.gather(
#         get_groq_response(),
#         get_gemini_response()
#     )
    
#     return (
#         groq_response.choices[0].message.content,
#         gemini_response.choices[0].message.content
#     )

# # Sidebar configuration
# with st.sidebar:
#     st.header("📁 Configuration")
    
#     groq_api_key = st.text_input("🔑 Groq API Key:", type="password")
#     gemini_api_key = st.text_input("🔑 Gemini API Key:", type="password")
    
#     if groq_api_key and st.session_state.groq_client is None:
#         st.session_state.groq_client = Groq(api_key=groq_api_key)
    
#     uploaded_file = st.file_uploader("📤 Upload Document (PDF)", type=["pdf"])
    
#     if uploaded_file and groq_api_key and gemini_api_key:
#         # Calculate file hash to avoid reprocessing
#         file_content = uploaded_file.getvalue()
#         current_hash = hash(file_content)
        
#         if current_hash != st.session_state.document_hash:
#             with st.spinner("Processing document..."):
#                 st.session_state.vector_store = asyncio.run(
#                     process_document(file_content, gemini_api_key)
#                 )
#                 st.session_state.document_hash = current_hash
#                 st.success("Document processed successfully!")
    
#     if st.button("Clear Chat History"):
#         clear_chat_history()

# # Main interface
# st.title("📚 Document Chat Comparison 🤖")

# # Model selection
# col1, col2 = st.columns(2)
# with col1:
#     groq_model = st.selectbox(
#         "Select Groq Model:",
#         ["mixtral-8x7b-32768", "llama-3.3-70b-versatile", "llama3-8b-8192"]
#     )
# with col2:
#     gemini_model = st.selectbox(
#         "Select Gemini Model:",
#         ["gemini-2.0-flash-exp", "gemini-1.5-flash-8b", "gemini-1.5-pro"]
#     )

# # Chat interface
# chat_col1, chat_col2 = st.columns(2)

# with chat_col1:
#     st.subheader("Groq Response")
#     for message in st.session_state.messages_groq:
#         with st.chat_message(message["role"]):
#             st.markdown(message["content"])

# with chat_col2:
#     st.subheader("Gemini Response")
#     for message in st.session_state.messages_gemini:
#         with st.chat_message(message["role"]):
#             st.markdown(message["content"])

# # Chat input processing
# if prompt := st.chat_input("Ask a question about your document"):
#     if not st.session_state.vector_store:
#         st.error("Please upload a document first!")
#     elif not groq_api_key or not gemini_api_key:
#         st.error("Please provide both API keys!")
#     else:
#         # Add user messages
#         st.session_state.messages_groq.append({"role": "user", "content": prompt})
#         st.session_state.messages_gemini.append({"role": "user", "content": prompt})
        
#         # Get context
#         retriever = st.session_state.vector_store.as_retriever()
#         context = asyncio.run(retriever.ainvoke(prompt))
#         context_text = "\n".join([doc.page_content for doc in context])
        
#         # Prepare prompt
#         prompt_template = PromptTemplate(
#             input_variables=["context", "question"],
#             template="""
#             Use the following context to answer the question. If the answer cannot be found in the context, say so.
#             The context is in markdown format, so please format your response accordingly.
            
#             Context: {context}
            
#             Question: {question}
            
#             Answer:
#             """
#         )
        
#         formatted_prompt = prompt_template.format(context=context_text, question=prompt)
        
#         try:
#             # Get responses concurrently
#             groq_content, gemini_content = asyncio.run(
#                 get_model_responses(
#                     formatted_prompt,
#                     st.session_state.groq_client,
#                     groq_model,
#                     gemini_model,
#                     gemini_api_key
#                 )
#             )
            
#             # Update chat histories
#             st.session_state.messages_groq.append({"role": "assistant", "content": groq_content})
#             st.session_state.messages_gemini.append({"role": "assistant", "content": gemini_content})
            
#             st.rerun()
            
#         except Exception as e:
#             st.error(f"Error generating response: {str(e)}")

import os
import streamlit as st
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_core.prompts import PromptTemplate
import tempfile
from litellm import completion
import pymupdf4llm
import asyncio
from groq import Groq
from functools import lru_cache

# Page configuration
st.set_page_config(page_title="Document Chat Comparison", page_icon="🤖", layout="wide")

# Initialize session state with a dictionary comprehension
INITIAL_STATE = {
    "messages_groq": [],
    "messages_gemini": [],
    "vector_store": None,
    "groq_client": None,
    "document_hash": None  # Add hash to check if document has changed
}

for key, value in INITIAL_STATE.items():
    if key not in st.session_state:
        st.session_state[key] = value

@lru_cache(maxsize=1)
def get_text_splitter():
    """Cache the text splitter instance"""
    return RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200,
        separators=["\n\n", "\n", " ", ""]
    )

def clear_chat_history():
    st.session_state.messages_groq = []
    st.session_state.messages_gemini = []
    st.rerun()

async def process_document(file_content, gemini_api_key, file_extension):
    """Process document asynchronously"""
    if file_extension == 'pdf':
        # PDF processing logic
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as temp_file:
            temp_file.write(file_content)
            temp_file_path = temp_file.name

        try:
            markdown_text = pymupdf4llm.to_markdown(temp_file_path)
            text_splitter = get_text_splitter()
            texts = text_splitter.create_documents([markdown_text])
            
            embeddings = GoogleGenerativeAIEmbeddings(
                model="models/embedding-001",
                google_api_key=gemini_api_key
            )
            
            vector_store = FAISS.from_documents(
                documents=texts,
                embedding=embeddings
            )
            return vector_store
        
        finally:
            if os.path.exists(temp_file_path):
                os.remove(temp_file_path)
                
    elif file_extension == 'txt':
        # TXT processing logic
        text = file_content.decode('utf-8')
        text_splitter = get_text_splitter()
        texts = text_splitter.create_documents([text])
        
        embeddings = GoogleGenerativeAIEmbeddings(
            model="models/embedding-001",
            google_api_key=gemini_api_key
        )
        
        vector_store = FAISS.from_documents(
            documents=texts,
            embedding=embeddings
        )
        return vector_store

async def get_model_responses(formatted_prompt, groq_client, groq_model, gemini_model, gemini_api_key):
    """Get responses from both models concurrently"""
    async def get_groq_response():
        return groq_client.chat.completions.create(
            model=groq_model,
            messages=[{"role": "user", "content": formatted_prompt}]
        )
    
    async def get_gemini_response():
        return completion(
            model=f"gemini/{gemini_model}",
            messages=[{"role": "user", "content": formatted_prompt}],
            api_key=gemini_api_key
        )
    
    groq_response, gemini_response = await asyncio.gather(
        get_groq_response(),
        get_gemini_response()
    )
    
    return (
        groq_response.choices[0].message.content,
        gemini_response.choices[0].message.content
    )

# Sidebar configuration
with st.sidebar:
    st.header("📁 Configuration")
    
    groq_api_key = st.text_input("🔑 Groq API Key:", type="password")
    gemini_api_key = st.text_input("🔑 Gemini API Key:", type="password")
    
    if groq_api_key and st.session_state.groq_client is None:
        st.session_state.groq_client = Groq(api_key=groq_api_key)
    
    uploaded_file = st.file_uploader("📤 Upload Document (PDF/TXT)", type=["pdf", "txt"])
    
    if uploaded_file and groq_api_key and gemini_api_key:
        # Calculate file hash to avoid reprocessing
        file_content = uploaded_file.getvalue()
        current_hash = hash(file_content)
        file_extension = uploaded_file.name.split('.')[-1].lower() if '.' in uploaded_file.name else ''
        
        if current_hash != st.session_state.document_hash:
            with st.spinner("Processing document..."):
                st.session_state.vector_store = asyncio.run(
                    process_document(file_content, gemini_api_key, file_extension)
                )
                st.session_state.document_hash = current_hash
                st.success("Document processed successfully!")
    
    if st.button("Clear Chat History"):
        clear_chat_history()

# Main interface
st.title("📚 Document Chat Comparison 🤖")

# Model selection
col1, col2 = st.columns(2)
with col1:
    groq_model = st.selectbox(
        "Select Groq Model:",
        ["llama-3.3-70b-versatile", "mixtral-8x7b-32768", "llama3-8b-8192"]
    )
with col2:
    gemini_model = st.selectbox(
        "Select Gemini Model:",
        ["gemini-2.0-flash-exp", "gemini-1.5-flash-8b", "gemini-1.5-pro"]
    )

# Chat interface
chat_col1, chat_col2 = st.columns(2)

with chat_col1:
    st.subheader("Groq Response")
    for message in st.session_state.messages_groq:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

with chat_col2:
    st.subheader("Gemini Response")
    for message in st.session_state.messages_gemini:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

# Chat input processing
if prompt := st.chat_input("Ask a question about your document"):
    if not st.session_state.vector_store:
        st.error("Please upload a document first!")
    elif not groq_api_key or not gemini_api_key:
        st.error("Please provide both API keys!")
    else:
        # Add user messages
        st.session_state.messages_groq.append({"role": "user", "content": prompt})
        st.session_state.messages_gemini.append({"role": "user", "content": prompt})
        
        # Get context
        retriever = st.session_state.vector_store.as_retriever()
        context = asyncio.run(retriever.ainvoke(prompt))
        context_text = "\n".join([doc.page_content for doc in context])
        
        # Prepare prompt
        prompt_template = PromptTemplate(
            input_variables=["context", "question"],
            template="""
            Use the following context to answer the question. If the answer cannot be found in the context, say so.
            The context is in markdown format, so please format your response accordingly.
            
            Context: {context}
            
            Question: {question}
            
            Answer:
            """
        )
        
        formatted_prompt = prompt_template.format(context=context_text, question=prompt)
        
        try:
            # Get responses concurrently
            groq_content, gemini_content = asyncio.run(
                get_model_responses(
                    formatted_prompt,
                    st.session_state.groq_client,
                    groq_model,
                    gemini_model,
                    gemini_api_key
                )
            )
            
            # Update chat histories
            st.session_state.messages_groq.append({"role": "assistant", "content": groq_content})
            st.session_state.messages_gemini.append({"role": "assistant", "content": gemini_content})
            
            st.rerun()
            
        except Exception as e:
            st.error(f"Error generating response: {str(e)}")
