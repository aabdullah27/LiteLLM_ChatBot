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

# Page configuration
st.set_page_config(page_title="Document Chat Comparison", page_icon="🤖", layout="wide")

# Initialize session state for chat history
if "messages_groq" not in st.session_state:
    st.session_state.messages_groq = []
if "messages_gemini" not in st.session_state:
    st.session_state.messages_gemini = []
if "vector_store" not in st.session_state:
    st.session_state.vector_store = None
if "groq_client" not in st.session_state:
    st.session_state.groq_client = None

# Function to clear chat history
def clear_chat_history():
    st.session_state.messages_groq = []
    st.session_state.messages_gemini = []
    st.rerun()

# Sidebar for configuration
with st.sidebar:
    st.header("📁 Configuration")
    
    # API Keys
    groq_api_key = st.text_input("🔑 Groq API Key:", type="password")
    gemini_api_key = st.text_input("🔑 Gemini API Key:", type="password")
    
    # Initialize Groq client if API key is provided
    if groq_api_key:
        st.session_state.groq_client = Groq(api_key=groq_api_key)
    
    # Document upload
    uploaded_file = st.file_uploader("📤 Upload Document (PDF)", type=["pdf"])
    
    if uploaded_file and groq_api_key and gemini_api_key:
        with st.spinner("Processing document..."):
            # Save uploaded file temporarily
            with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as temp_file:
                temp_file.write(uploaded_file.getvalue())
                temp_file_path = temp_file.name

            try:
                # Convert PDF to markdown using pymupdf4llm
                markdown_text = pymupdf4llm.to_markdown(temp_file_path)
                
                # Split text
                text_splitter = RecursiveCharacterTextSplitter(
                    chunk_size=1000,
                    chunk_overlap=200,
                    separators=["\n\n", "\n", " ", ""]
                )
                
                # Create text chunks
                texts = text_splitter.create_documents([markdown_text])
                
                # Create vector store with Google embeddings
                embeddings = GoogleGenerativeAIEmbeddings(
                    model="models/embedding-001",
                    google_api_key=gemini_api_key
                )
                st.session_state.vector_store = FAISS.from_documents(
                    documents=texts,
                    embedding=embeddings
                )
                
                st.success("Document processed successfully!")
            
            finally:
                if os.path.exists(temp_file_path):
                    os.remove(temp_file_path)
    
    # Clear chat button
    st.header("🧹 Clear Chat")
    if st.button("Clear Chat History"):
        clear_chat_history()

# Main chat interface
st.title("📚 Document Chat Comparison 🤖")

# Model selection dropdowns in two columns
col1, col2 = st.columns(2)
with col1:
    groq_model = st.selectbox(
        "Select Groq Model:",
        ["mixtral-8x7b-32768", "llama-3.3-70b-versatile", "llama3-8b-8192"]
    )
with col2:
    gemini_model = st.selectbox(
        "Select Gemini Model:",
        ["gemini-2.0-flash-exp", "gemini-1.5-flash-8b", "gemini-1.5-pro"]
    )

# Chat interface split into two columns
chat_col1, chat_col2 = st.columns(2)

# Display chat histories
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

# Chat input
if prompt := st.chat_input("Ask a question about your document"):
    if not st.session_state.vector_store:
        st.error("Please upload a document first!")
    elif not groq_api_key or not gemini_api_key:
        st.error("Please provide both API keys!")
    else:
        # Add user message to both chat histories
        st.session_state.messages_groq.append({"role": "user", "content": prompt})
        st.session_state.messages_gemini.append({"role": "user", "content": prompt})
        
        # Retrieve relevant context using the new invoke method
        retriever = st.session_state.vector_store.as_retriever()
        context = asyncio.run(retriever.ainvoke(prompt))  # Using ainvoke for async retrieval
        context_text = "\n".join([doc.page_content for doc in context])
        
        # Prepare prompt template
        template = """
        Use the following context to answer the question. If the answer cannot be found in the context, say so.
        The context is in markdown format, so please format your response accordingly.
        
        Context: {context}
        
        Question: {question}
        
        Answer:
        """
        
        prompt_template = PromptTemplate(
            input_variables=["context", "question"],
            template=template
        )
        
        formatted_prompt = prompt_template.format(context=context_text, question=prompt)
        
        try:
            # Get Groq response using the Groq client
            groq_chat_completion = st.session_state.groq_client.chat.completions.create(
                model=groq_model,
                messages=[{"role": "user", "content": formatted_prompt}]
            )
            groq_content = groq_chat_completion.choices[0].message.content
            st.session_state.messages_groq.append({"role": "assistant", "content": groq_content})
            
            # Get Gemini response
            gemini_response = completion(
                model=f"gemini/{gemini_model}",
                messages=[{"role": "user", "content": formatted_prompt}],
                api_key=gemini_api_key
            )
            gemini_content = gemini_response.choices[0].message.content
            st.session_state.messages_gemini.append({"role": "assistant", "content": gemini_content})
            
            # Force refresh to show new messages
            st.rerun()
            
        except Exception as e:
            st.error(f"Error generating response: {str(e)}")
