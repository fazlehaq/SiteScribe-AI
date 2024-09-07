from langchain_community.embeddings import TensorflowHubEmbeddings
embeddings = None

def getEmbedder():
    global embeddings
    if embeddings is None :
        model_url = "https://tfhub.dev/google/universal-sentence-encoder/4"
        embeddings = TensorflowHubEmbeddings(model_url=model_url) 
        
    return embeddings