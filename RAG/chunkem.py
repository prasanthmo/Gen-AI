import os
import nltk
import chromadb
from sentence_transformers import SentenceTransformer
from nltk.tokenize import sent_tokenize

# Download necessary NLTK resources for sentence tokenization
nltk.download('punkt')
nltk.download('punkt_tab')

# Load the SentenceTransformer model for embedding
model = SentenceTransformer('all-MiniLM-L6-v2')  # Example of a lightweight transformer model

# Initialize ChromaDB client
client = chromadb.Client()

# Function to read text from a file
def read_text_file(file_path):
    with open(file_path, 'r', encoding='utf-8') as file:
        return file.read()

# Function for sentence-level chunking
def sentence_level_chunking(text):
    # Using nltk to split the text into sentences
    sentences = sent_tokenize(text)
    return sentences

# Function to embed and store chunks in ChromaDB
def embed_and_store_in_chromadb(text, collection_name="documents"):
    # Tokenize text into sentences
    chunks = sentence_level_chunking(text)
    
    # Initialize ChromaDB collection
    collection = client.get_or_create_collection(collection_name)

    # Embed chunks using the SentenceTransformer model
    embeddings = model.encode(chunks)

    # Store the embeddings and chunks in ChromaDB
    for i, chunk in enumerate(chunks):
        collection.add(
            documents=[chunk],  # Text chunk
            metadatas=[{"text_chunk": chunk}],  # Metadata (optional, could be any additional information)
            embeddings=[embeddings[i].tolist()],  # Embedding vector
            ids=[str(i)]  # Unique identifier for each chunk
        )

    print(f"Stored {len(chunks)} chunks in ChromaDB.")

# Example usage
if __name__ == "__main__":
    # Provide the actual path to your text document
    file_path = r"C:\Users\mohan\OneDrive\Desktop\GenAI\pdf\sample3.txt"  # Corrected file path
    
    # Read the text document
    text = read_text_file(file_path)
    
    # Embed and store chunks in ChromaDB
    embed_and_store_in_chromadb(text)
