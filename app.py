from langchain_core.messages import HumanMessage, AIMessage 
from embedder import getEmbedder
from embedd_files import embeddFiles
from chain_setup import getRagChain


# assuming files are present in books directory
# Taking filenames as input
files = []
while True :
    print("1) Add file  \n2) Stop")
    choice = int(input())
    if choice == 2 : 
        break
    filename = input("filename : ")
    files.append(filename)

if len(files) == 0 :
    raise Exception("At least 1 file is expected")

    
# Taking collection name as input
COLLECTION_NAME = input("Enter Name Of Vector Store")

if len(COLLECTION_NAME) == 0 :
    raise Exception("Vector store name expected")


# embedding input file and storing in vector store
embeddFiles(files,COLLECTION_NAME)
rag_chain =getRagChain(COLLECTION_NAME)



# Function to simulate a continual chat
def continual_chat():
    print("Start chatting with the AI! Type 'exit' to end the conversation.")
    chat_history = []  # Collect chat history here (a sequence of messages)
    while True:
        print(chat_history)  # Print the chat history to debug
        query = input("You: ")
        if query.lower() == "exit":
            break
        
        # Process the user's query through the retrieval chain
        result = rag_chain.invoke({"input": query, "chat_history": chat_history})
        # Display the AI's response
        print(f"AI: {result['answer']}")
        
        # Update the chat history
        chat_history.append(HumanMessage(content=query))
        chat_history.append(AIMessage(content=result["answer"]))  # Corrected here

        # Debugging output for chat history
        print("Updated chat history:", chat_history)

continual_chat()