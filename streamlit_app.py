
import os
import streamlit as st
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import PromptTemplate
from langchain_community.document_loaders import YoutubeLoader
from langchain_core.runnables import RunnableParallel, RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# --- Helper Functions (from your original script) ---

def get_youtube_docs(url):
    """Loads transcript from a YouTube URL."""
    loader = YoutubeLoader.from_youtube_url(url, add_video_info=False)
    documents = loader.load()
    return documents

def split_docs(documents):
    """Splits documents into chunks."""
    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    chunks = splitter.split_documents(documents)
    return chunks

def create_vector_store(chunks):
    """Creates a FAISS vector store from chunks."""
    embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
    vector_store = FAISS.from_documents(chunks, embeddings)
    return vector_store

def create_rag_chain(retriever):
    """Creates the RAG chain for question answering."""
    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.2)
    
    prompt_template = PromptTemplate(
        template='''
      You are a helpful assistant.
      Answer ONLY from the provided transcript context.
      If the context is insufficient, just say you don't know.

      {context}
      Question: {question}
    ''',
        input_variables=['context', 'question']
    )

    def format_docs(docs):
        return "\n\n".join(doc.page_content for doc in docs)

    rag_chain = (
        RunnableParallel(
            context=retriever | format_docs,
            question=RunnablePassthrough()
        )
        | prompt_template
        | llm
        | StrOutputParser()
    )
    return rag_chain

# --- Streamlit App ---

st.set_page_config(page_title="YouTube Video Q&A", layout="wide", initial_sidebar_state="expanded")

st.title("🎬 YouTube Video Q&A")
st.markdown(
    """
    <style>
    .stApp {
        background-color: #f0f2f6;
        color: black;
    }
    .stButton>button {
        background-color: #4CAF50;
        color: black;
    }
    .answer-box {
        background-color: #e8e8e8;
        padding: 15px;
        border-radius: 5px;
        border: 1px solid #ccc;
        color: black;
    }
    </style>
    """, unsafe_allow_html=True)

# --- Sidebar for API Key and Inputs ---
with st.sidebar:
    
    # Check for API key in environment, otherwise ask for it
    openai_api_key = os.getenv("OPENAI_API_KEY")
    if not openai_api_key:
        openai_api_key = st.text_input("Enter your OpenAI API Key:", type="password")

    st.header("Inputs")
    youtube_url = st.text_input("Enter YouTube URL:")
    question = st.text_area("Ask a question about the video:")
    analyze_button = st.button("Analyze Video")

# --- Main Content Area ---
if analyze_button:
    if not openai_api_key:
        st.error("Please provide your OpenAI API Key in the sidebar.")
    elif not youtube_url:
        st.error("Please enter a YouTube URL.")
    elif not question:
        st.error("Please enter a question.")
    else:
        try:
            # Set the API key for the session
            os.environ["OPENAI_API_KEY"] = openai_api_key
            
            with st.spinner("Processing... Please wait."):
                # 1. Load Documents
                docs = get_youtube_docs(youtube_url)
                if not docs:
                    st.error("Could not load transcript. Please check the URL and try again.")
                    st.stop() 
                   

                # 2. Split Documents
                chunks = split_docs(docs)
                if not chunks:
                    st.error("Failed to split the document.")
                    st.stop()

                # 3. Create Vector Store
                vector_store = create_vector_store(chunks)
                retriever = vector_store.as_retriever(search_type="similarity", search_kwargs={"k": 4})

                # 4. Create RAG Chain and Get Answer
                rag_chain = create_rag_chain(retriever)
                answer = rag_chain.invoke(question)

            st.subheader("Answer:")
            st.markdown(f'<div class="answer-box">{answer}</div>', unsafe_allow_html=True)

        except Exception as e:
            st.error(f"An error occurred: {e}")

else:
    st.info("Enter a YouTube URL and a question in the sidebar and click 'Analyze Video'.")
