import os
from langchain.text_splitter import CharacterTextSplitter
from langchain_community.document_loaders import TextLoader
from langchain_community.vectorstores import Chroma
from embedder import getEmbedder


def embeddFiles(filenames : list[str],collection_name : str) -> None :
    print("In embedd file")
    current_dir = os.path.dirname(os.path.abspath(__file__))
    persistent_directory = os.path.join(current_dir,"db",collection_name)

    print(os.path.join(current_dir,"db",collection_name) , persistent_directory)
    # Checking if the chroma vector already exists
    if os.path.exists(persistent_directory) :
        print("Vector store already exists. No need to initialize.")
        return None
    
    # Ensuring all given files exists
    for i in range(len(filenames)):
        curr_file_path = os.path.join(current_dir,"books",filenames[i])
        if not os.path.exists(curr_file_path) :
            print(f"File {filenames[i]} does not exists please check!")
            return None
    
    # Combining all given docs
    docs = []
    for i in range(len(filenames)):
        curr_file_path = os.path.join(current_dir,"books",filenames[i])
        loader = TextLoader(curr_file_path)
        book_docs = loader.load()
        for doc in book_docs :
            # Add metadata to each document indicating its source
            doc.metadata = {"source": filenames[i]}
            docs.append(doc)

    # Split the combined document in chunks
    text_splitter = CharacterTextSplitter(chunk_size=1000,chunk_overlap=0)
    docs = text_splitter.split_documents(docs)
    
    # Displying chunks 
    print("\n--- Document Chunks Information ---")
    print(f"Number of document chunks : {len(docs)}")
    print(f"First Chunk: \n\t{docs[0].page_content}")
    print(f"Last Chunk: \n\t{docs[len(docs)-1].page_content}")

    # Create Embeddings
    print("\n--- Creating Embeddings ---")
    embeddings = getEmbedder()
    print("\n--- Finished creating embeddings ---")

    # Create the vector store and persist it automatically
    print("\n--- Creating vector store ---")
    db = Chroma.from_documents(
        docs, embeddings, persist_directory=persistent_directory
    )
    print("\n--- Finished creating vector store ---")

