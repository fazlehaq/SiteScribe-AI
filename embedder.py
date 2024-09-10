# from langchain_community.embeddings import TensorflowHubEmbeddings
# embeddings = None
# def getEmbedder():
#     global embeddings
#     if embeddings is None :
#         model_url = "https://tfhub.dev/google/universal-sentence-encoder/4"
#         embeddings = TensorflowHubEmbeddings(model_url=model_url) 
#     return embeddings

from dotenv import load_dotenv
from langchain_google_genai import GoogleGenerativeAIEmbeddings
embeddings = None
load_dotenv()

def getEmbedder():
    global embeddings
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    return embeddings