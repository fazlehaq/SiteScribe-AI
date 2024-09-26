import os
from dotenv import load_dotenv
from langchain_chroma import Chroma
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.runnables import RunnableBranch , RunnableSerializable , RunnableLambda , RunnableParallel
from langchain_core.output_parsers import StrOutputParser

from embedder import getEmbedder

load_dotenv()

def getRagChain(collection_name : str) -> RunnableSerializable :
    current_dir = os.path.dirname(os.path.abspath(__file__))
    persistent_directory = os.path.join(current_dir, "db", collection_name)

    # Define the embedding model
    embeddings = getEmbedder()

    # Load the existing vector store with the embedding function
    db = Chroma(persist_directory=persistent_directory, embedding_function=embeddings)

    # Create a retriever for querying the vector store
    retriever = db.as_retriever(
        search_type="similarity_score_threshold",
        search_kwargs={"k": 3 , "score_threshold" : 0.3},
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
            ("human", "{query}"),
        ]
    )

    # Create a history-aware retriever
    # history_aware_retriever = create_history_aware_retriever(
    #     llm, retriever, contextualize_q_prompt
    # )

    # Answer question prompt
    qa_system_prompt = (
        "You are an assistant for question-answering tasks. Use "
        "the below pieces of retrieved context to answer the "
        "question , Dont use your knowledge base to answer"
        "Your answer should be purely based on below provided context"
        " If you cant find the answer in the provided prompt/context, just say that you "
        "don't know. Use three sentences maximum and keep the answer "
        "concise."
        "Here is the context find answer from it"
        "\n\n"
        "{context}"
        
    )


    #Create a prompt template for answering questions
    qa_prompt = ChatPromptTemplate.from_messages(
        [
            ("system", qa_system_prompt),
            MessagesPlaceholder("chat_history"),
            ("human", "{query}"),
        ]
    )

    # PreBuilt History aware retriver in langchain
    # Create a chain to combine documents for question answering
    # question_answer_chain = create_stuff_documents_chain(llm, qa_prompt)
    # Create a retrieval chain that combines the history-aware retriever and the question answering chain
    # rag_chain = create_retrieval_chain(history_aware_retriever, question_answer_chain)


    # Lambda runnables
    asItIsRunnable = RunnableLambda(lambda x : x)
    printF = RunnableLambda(lambda x : (print(x) , x)[1] ) # for debugging
    getQuery = RunnableLambda(lambda x : x["query"])
    getContextualizedQPrompt = RunnableLambda(lambda x : contextualize_q_prompt.format_prompt(**x))
    retrieveRelDocs = RunnableLambda(lambda x :  retriever.invoke(x["query"]))
    getFormattedQAPrompt = RunnableLambda(lambda x : qa_prompt.format_prompt(**x))

    q_branches = RunnableBranch(
        (
            lambda x : len(x["chat_history"]) == 0,
            getQuery
        ),
            getContextualizedQPrompt | llm | printF |  StrOutputParser()
    )

    # My own history aware retrival chain
    rag_chain = (
        RunnableParallel(branches={ "question_rephrase" :q_branches , "as_it_is": asItIsRunnable}) # Rephrase the query from the chat history if exists
        | RunnableLambda( lambda x :  {"query" :  x ["branches"]["question_rephrase"] , "chat_history" : x["branches"]["as_it_is"]["chat_history"] }) # correctly structuring the input to the next chain
        | RunnableParallel(branches = { "retriever" : retrieveRelDocs ,"as_it_is" : asItIsRunnable }) # Retrieving the relevent docs
        | RunnableLambda ( lambda x : {"context" : x["branches"]["retriever"] , "query" : x["branches"]["as_it_is"]["query"] , "chat_history": x["branches"]["as_it_is"]["chat_history"] }) # combining inputs from last branch 
        | getFormattedQAPrompt # inserting query and retrived context from the vector store to the propmpt template
        | llm # calling llm and passing it the prompt
    )

    return rag_chain