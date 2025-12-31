import streamlit as st
from pypdf import PdfReader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
import os

def get_pdf_text(pdf_docs):
    """
    Extracts text from uploaded PDF files.
    """
    text = ""
    for pdf in pdf_docs:
        pdf_reader = PdfReader(pdf)
        for page in pdf_reader.pages:
            text += page.extract_text()
    return text

def get_text_chunks(text):
    """
    Splits the text into chunks using RecursiveCharacterTextSplitter.
    """
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200,
        length_function=len
    )
    chunks = text_splitter.split_text(text)
    return chunks

def get_vector_store(text_chunks, api_key):
    """
    Creates a FAISS vector store from text chunks using OpenAI embeddings.
    """
    embeddings = OpenAIEmbeddings(openai_api_key=api_key)
    vectorstore = FAISS.from_texts(texts=text_chunks, embedding=embeddings)
    return vectorstore

def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)

def main():
    st.set_page_config(page_title="DocuMind AI", page_icon="📚")
    st.header("📚 DocuMind AI - Chat with your PDF")

    # Sidebar for configuration
    with st.sidebar:
        st.title("Settings")
        api_key = st.text_input("Enter OpenAI API Key:", type="password")
        if not api_key:
            st.warning("Please enter your OpenAI API Key to proceed.")
        
        st.subheader("Your Documents")
        pdf_docs = st.file_uploader("Upload your PDFs here and click on 'Process'", accept_multiple_files=True, type="pdf")
        
        if st.button("Process"):
            if not api_key:
                st.error("Please enter an API Key first.")
            elif not pdf_docs:
                st.error("Please upload at least one PDF file.")
            else:
                with st.spinner("Processing..."):
                    # 1. Get PDF text
                    raw_text = get_pdf_text(pdf_docs)
                    
                    # 2. Get Text Chunks
                    text_chunks = get_text_chunks(raw_text)
                    
                    # 3. Create Vector Store
                    vectorstore = get_vector_store(text_chunks, api_key)
                    st.session_state.vectorstore = vectorstore
                    
                    st.success("Done! You can now ask questions about your documents.")

    # Main Area: Q&A
    user_question = st.text_input("Ask a question about your PDF:")

    if user_question:
        if not api_key:
             st.error("Please enter your API Key in the sidebar.")
        elif "vectorstore" not in st.session_state:
            st.warning("Please upload and process a PDF file first.")
        else:
            with st.spinner("Thinking..."):
                # Retrieve vector store from session state
                vectorstore = st.session_state.vectorstore
                
                # Create LLM
                llm = ChatOpenAI(temperature=0, openai_api_key=api_key, model_name="gpt-3.5-turbo")
                
                # Create LCEL Chain
                # This replaces the legacy RetrievalQA chain with a modern, transparent chain
                
                template = """Answer the question based only on the following context:
{context}

Question: {question}
"""
                prompt = ChatPromptTemplate.from_template(template)
                retriever = vectorstore.as_retriever()
                
                rag_chain = (
                    {"context": retriever | format_docs, "question": RunnablePassthrough()}
                    | prompt
                    | llm
                    | StrOutputParser()
                )
                
                response = rag_chain.invoke(user_question)
                st.write(response)

if __name__ == "__main__":
    main()
