# Building a Retrieval-Augmented Generation (RAG) System with LangChain
import os
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import PromptTemplate
from langchain_community.document_loaders import YoutubeLoader
from langchain_core.runnables import RunnableParallel, RunnablePassthrough, RunnableLambda
from langchain_core.output_parsers import StrOutputParser

# It's a good practice to load API keys from environment variables
# For example, using python-dotenv:
from dotenv import load_dotenv
load_dotenv()
# Make sure your OPENAI_API_KEY is set in your environment or a .env file
if not os.getenv("OPENAI_API_KEY"):
    print("Error: OPENAI_API_KEY environment variable not set.")
    exit()

def main():
    """Main function to run the RAG pipeline."""

    # Step 1a: Document Ingestion
    print("--- Loading YouTube transcript... ---")
    video_url = "https://www.youtube.com/watch?v=Hu4Yvq-g7_Y"
    loader = YoutubeLoader.from_youtube_url(video_url, add_video_info=False)
    documents = loader.load()

    if not documents:
        print("Could not load documents from the YouTube URL.")
        return

    # Step 1b: Text Splitting
    print("--- Splitting document into chunks... ---")
    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    chunks = splitter.split_documents(documents)

    if not chunks:
        print("Could not split documents into chunks.")
        return
    
    print(f"Document split into {len(chunks)} chunks.")

    # Step 1c & 1d: Embedding and Storing in Vector Store
    print("--- Creating vector store... ---")
    embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
    vector_store = FAISS.from_documents(chunks, embeddings)

    # Step 2: Retrieval
    retriever = vector_store.as_retriever(search_type="similarity", search_kwargs={"k": 4})

    # Step 3 & 4: Augmentation and Generation (Manual method)
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

    print("\n--- Summarization pf your videp... ---")
    summary_question = "Can you summarize the video?, mention important points "
    summary_retrieved_docs = retriever.invoke(summary_question)
    summary_context_text = "\n\n".join(doc.page_content for doc in summary_retrieved_docs)
    summary_final_prompt = prompt_template.invoke({"context": summary_context_text, "question": summary_question})
    summary_answer_manual = llm.invoke(summary_final_prompt)
    print(f"Question: {summary_question}")
    print(f"Answer: {summary_answer_manual.content}")
    print("-" * 20)

    # Building and using a LangChain Expression Language (LCEL) chain
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

if __name__ == "__main__":
    main()
