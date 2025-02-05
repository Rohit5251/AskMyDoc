import streamlit as st
from langchain_community.llms import Ollama
from pdf2image import convert_from_bytes
from PIL import Image
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import OllamaEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains import create_retrieval_chain
import tempfile    

import os
from dotenv import load_dotenv
load_dotenv()

os.environ['GOOGLE_API_KEY'] = os.getenv('GOOGLE_API_KEY')

st.title("AskMyDoc ")
st.subheader("Upload your PDF and get instant answers to your queries using AI-powered document analysis.")
op = st.sidebar.selectbox(label="Select here", options=['Gemini', 'Deepseek'])
if op == 'Gemini':
    st.write("Using Gemini...")
else:
    st.write("Using Deepseek...")

st.sidebar.header("Add your document")
file = st.sidebar.file_uploader(label="Upload your .pdf file", type=['pdf'])

if file:
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as temp_file:
        temp_file.write(file.read())
        temp_file_path = temp_file.name
    st.sidebar.success("Ready to chat!")
    # PDF preview
    st.sidebar.write("Your uploaded file is:")
    with open(temp_file_path, "rb") as f:
        images = convert_from_bytes(f.read())
    for i, image in enumerate(images):
        st.sidebar.image(image, caption=f"Page {i + 1}", use_column_width=True)

    # Load PDF
    loader = PyPDFLoader(temp_file_path)
    docs = loader.load()

    # Text Split
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    documents = text_splitter.split_documents(docs)

    if op == 'Deepseek':
        # Embedding
        db1 = FAISS.from_documents(documents, OllamaEmbeddings(model="deepseek-r1:1.5b"))
        llm = Ollama(
            model="deepseek-r1:1.5b",
        )
    else:
        # Embedding
        db1 = FAISS.from_documents(documents, GoogleGenerativeAIEmbeddings(model="models/embedding-001"))
        llm = ChatGoogleGenerativeAI(
            model="gemini-1.5-pro",
        )

    prompt = ChatPromptTemplate.from_template("""
    Answer the following question based only on the provided context:
    <context>
    {context}
    </context>
    Question: {input}
    """)

    # Stuff document chain
    document_chain = create_stuff_documents_chain(llm, prompt)

    # Retriever
    retriever = db1.as_retriever()

    # Retriever chain
    retrieval_chain = create_retrieval_chain(retriever, document_chain)

    # Initialize session state to store chat history
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []

    # Function to handle user input
    def handle_user_input():
        user_input = st.session_state.user_input
        if user_input:
            # Add user's message to chat history
            st.session_state.chat_history.append({"sender": "User", "message": user_input})
            # Generate response using retrieval chain
            res = retrieval_chain.invoke({"input": user_input})
            response = res['answer']
            st.session_state.chat_history.append({"sender": "Gemini", "message": response})
            # Clear input field
            st.session_state.user_input = ""

    for chat in st.session_state.chat_history:
        if chat["sender"] == "User":
            st.markdown(f"**You:** {chat['message']}")
        else:
            st.markdown(f"**AI:** {chat['message']}")

    # Input field at the bottom
    que=st.text_input(
        "Ask a question:",
        key="user_input",
        on_change=handle_user_input,
        placeholder="Type your question here..."
    )
    response = retrieval_chain.invoke({"input": que})
    res = response['answer']

    # Ensure </think> exists before splitting
    if "</think>" in res:
        filtered_response = res.split("</think>")[-1].strip()
    else:
        filtered_response = res  # Use the full response if </think> is not found

    st.write(filtered_response)
else:
    st.sidebar.warning("Please upload a PDF file to start.")
