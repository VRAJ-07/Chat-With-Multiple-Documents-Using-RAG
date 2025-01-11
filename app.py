import streamlit as st
import faiss
import os
from io import BytesIO
from docx import Document
import numpy as np
from langchain_community.document_loaders import WebBaseLoader
from PyPDF2 import PdfReader
from langchain.chains import RetrievalQA
from langchain.text_splitter import CharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_community.docstore.in_memory import InMemoryDocstore
from langchain_huggingface import HuggingFaceEndpoint
from dotenv import load_dotenv

load_dotenv()

HUGGINGFACE_API_KEY = os.environ['HUGGINGFACE_API_KEY']

def process_input(input_type, input_data):
    """Processes different input types and returns a vectorstore."""
    documents = ""
    if input_type == "Link":
        loader = WebBaseLoader(input_data)
        web_documents = loader.load()
        text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
        texts = [str(doc.page_content) for doc in text_splitter.split_documents(web_documents)]
    elif input_type == "PDF":
        for file in input_data:
            pdf_reader = PdfReader(BytesIO(file.read()))
            for page in pdf_reader.pages:
                documents += page.extract_text()
    elif input_type == "Text":
        documents = input_data
    elif input_type == "DOCX":
        for file in input_data:
            doc = Document(BytesIO(file.read()))
            documents += "\n".join([para.text for para in doc.paragraphs])
    elif input_type == "TXT":
        for file in input_data:
            documents += file.read().decode('utf-8')
    else:
        raise ValueError("Unsupported input type")

    if input_type != "Link":
        text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
        texts = text_splitter.split_text(documents)

    model_name = "sentence-transformers/all-mpnet-base-v2"
    model_kwargs = {'device': 'cpu'}
    encode_kwargs = {'normalize_embeddings': False}

    hf_embeddings = HuggingFaceEmbeddings(
        model_name=model_name,
        model_kwargs=model_kwargs,
        encode_kwargs=encode_kwargs
    )
    # Create FAISS index
    sample_embedding = np.array(hf_embeddings.embed_query("sample text"))
    dimension = sample_embedding.shape[0]
    index = faiss.IndexFlatL2(dimension)
    # Create FAISS vector store with the embedding function
    vector_store = FAISS(
        embedding_function=hf_embeddings.embed_query,
        index=index,
        docstore=InMemoryDocstore(),
        index_to_docstore_id={},
    )
    vector_store.add_texts(texts)  # Add documents to the vector store
    return vector_store

def answer_question(vectorstore, query):
    """Answers a question based on the provided vectorstore."""
    try:
        llm = HuggingFaceEndpoint(
            repo_id='mistralai/Mistral-7B-Instruct-v0.2',
            token=HUGGINGFACE_API_KEY,
            temperature=0.6,
            max_new_tokens=1000
        )
        qa = RetrievalQA.from_chain_type(llm=llm, retriever=vectorstore.as_retriever())
        result = qa({"query": query})
        return result.get("result", "No answer found. The model might not have understood the question.")
    except Exception as e:
        return f"An error occurred: {str(e)}"

def main():
    st.title("AskIt - Q&A with Multiple Files")
    st.write("Upload documents or provide a link to ask questions based on their content.")

    input_type = st.selectbox("Input Type", ["Link", "PDF", "Text", "DOCX", "TXT"])
    input_data = None

    if input_type == "Link":
        number_of_links = st.number_input("Number of Links", min_value=1, max_value=20, step=1)
        input_data = [st.text_input(f"Link {i+1}") for i in range(int(number_of_links))]
        input_data = [link for link in input_data if link.strip()]  # Remove empty links
    elif input_type == "Text":
        input_data = st.text_area("Enter the text")
    elif input_type in ["PDF", "DOCX", "TXT"]:
        input_data = st.file_uploader(
            f"Upload {input_type} files", type=[input_type.lower()], accept_multiple_files=True
        )

    if st.button("Process Input"):
        if input_type == "Link" and not input_data:
            st.error("Please provide at least one valid link.")
        elif input_type != "Link" and not input_data:
            st.error("Please upload at least one file or enter valid text.")
        else:
            try:
                vectorstore = process_input(input_type, input_data)
                st.session_state["vectorstore"] = vectorstore
                st.success("Input processed successfully!")
            except Exception as e:
                st.error(f"An error occurred while processing the input: {str(e)}")

    if "vectorstore" in st.session_state:
        query = st.text_input("Ask your question:")
        if st.button("Submit"):
            if query.strip():
                answer = answer_question(st.session_state["vectorstore"], query)
                st.markdown(f"**Answer:** {answer}")
                if "history" not in st.session_state:
                    st.session_state["history"] = []
                st.session_state["history"].append((query, answer))
            else:
                st.error("Please enter a valid question.")

        # Optionally, display question-answer history
        if "history" in st.session_state and st.session_state["history"]:
            with st.expander("Question-Answer History"):
                for i, (q, a) in enumerate(st.session_state["history"], 1):
                    st.markdown(f"**Q{i}:** {q}")
                    st.markdown(f"**A{i}:** {a}")

if __name__ == "__main__":
    main()
