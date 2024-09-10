from langchain_core.messages import HumanMessage, AIMessage 
from embedd_files import embeddFiles
from chain_setup import getRagChain

# Taking input filename and name of the vector store
files = []
COLLECTION_NAME = None

choice = int(input("1) Use existing vectore store\n2) Create new vectore store\n"))
if choice == 2 :

    while True :
        choice = int(input("1)Add Filename\n2)Stop\n"))
        if choice == 2 :
            break
        filename = input("Enter filename : ")
        files.append(filename)
    
    if len(files) == 0 :
        raise Exception("At least 1 file expected")

    COLLECTION_NAME = input("Enter name for your vector store :")

    if len(COLLECTION_NAME) == 0 :
        raise Exception("Vector store name is expected")

    embeddFiles(files,COLLECTION_NAME)    

else :
    COLLECTION_NAME = input("Enter name for your vector store ")
    
    if len(COLLECTION_NAME) == 0 :
        raise Exception("Vector store name is expected")
    
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
        result = rag_chain.invoke({"query": query, "chat_history": chat_history})
        # Display the AI's response
        print(f"AI: {result.content}")
        
        # Update the chat history
        chat_history.append(HumanMessage(content=query))
        chat_history.append(AIMessage(content= result.content ))  # Corrected here

        # Debugging output for chat history
        # print("Updated chat history:", chat_history)

continual_chat()