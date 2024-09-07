import os
from dotenv import load_dotenv
from langchain.chains import create_history_aware_retriever, create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_community.vectorstores import Chroma
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_google_genai import ChatGoogleGenerativeAI

from embedder import getEmbedder

load_dotenv()

def getRagChain(collection_name : str) -> any:
    current_dir = os.path.dirname(os.path.abspath(__file__))
    persistent_directory = os.path.join(current_dir, "db", collection_name)

    # Define the embedding model
    embeddings = getEmbedder()

    # Load the existing vector store with the embedding function
    db = Chroma(persist_directory=persistent_directory, embedding_function=embeddings)

    # Create a retriever for querying the vector store
    retriever = db.as_retriever(
        search_type="similarity",
        search_kwargs={"k": 3},
    )

    # Create the LLM model
    llm = ChatGoogleGenerativeAI(model="gemini-1.5-flash")



    # Contextualize question prompt
    contextualize_q_system_prompt = (
        "Given a chat history and the latest user question "
        "which might reference context in the chat history, "
        "formulate a standalone question which can be understood "
        "without the chat history. Do NOT answer the question, just "
        "reformulate it if needed and otherwise return it as is."
    )


    contextualize_q_prompt = ChatPromptTemplate.from_messages(
        [
            ("system", contextualize_q_system_prompt),
            MessagesPlaceholder("chat_history"),
            ("human", "{input}"),
        ]
    )

    # Create a history-aware retriever
    history_aware_retriever = create_history_aware_retriever(
        llm, retriever, contextualize_q_prompt
    )

    # Answer question prompt
    qa_system_prompt = (
        "You are an assistant for question-answering tasks. Use "
        "the below pieces of retrieved context to answer the "
        "question.Dont use any other knowledge source." 
        "Your answer should be purely based on below provided context"
        "Dont use your knowledge or web"
        " If you don't know the answer, just say that you "
        "don't know. Use three sentences maximum and keep the answer "
        "concise."
        "Here is the context find answer from it"
        "\n\n"
        "{context}"
        
    )


    # Create a prompt template for answering questions
    qa_prompt = ChatPromptTemplate.from_messages(
        [
            ("system", qa_system_prompt),
            MessagesPlaceholder("chat_history"),
            ("human", "{input}"),
        ]
    )

    # Create a chain to combine documents for question answering
    question_answer_chain = create_stuff_documents_chain(llm, qa_prompt)

    # Create a retrieval chain that combines the history-aware retriever and the question answering chain
    rag_chain = create_retrieval_chain(history_aware_retriever, question_answer_chain)

    return rag_chain
