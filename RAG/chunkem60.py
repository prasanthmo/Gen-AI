import os
import time
import difflib
import nltk
import chromadb
from watchdog.observers import Observer
from watchdog.events import FileSystemEventHandler
from sentence_transformers import SentenceTransformer
from nltk.tokenize import sent_tokenize

# Download necessary NLTK resources for sentence tokenization
nltk.download('punkt')  # Correct download for sentence tokenization

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
    sentences = sent_tokenize(text)  # Tokenize text into sentences
    return sentences

# Function to embed and store chunks in ChromaDB
def embed_and_store_in_chromadb(text, collection_name="documents"):
    chunks = sentence_level_chunking(text)  # Tokenize text into chunks (sentences)
    collection = client.get_or_create_collection(collection_name)  # Create or access ChromaDB collection

    embeddings = model.encode(chunks)  # Embed chunks using the SentenceTransformer model

    # Store the embeddings and chunks in ChromaDB
    for i, chunk in enumerate(chunks):
        collection.add(
            documents=[chunk],  # Text chunk
            metadatas=[{"text_chunk": chunk}],  # Metadata (optional)
            embeddings=[embeddings[i].tolist()],  # Embedding vector
            ids=[str(i)]  # Unique identifier for each chunk
        )

    print(f"Stored {len(chunks)} chunks in ChromaDB.")

# Function to detect added and deleted content
def detect_changes(old_text, new_text):
    # Split the text into lines to compare using difflib
    diff = difflib.ndiff(old_text.splitlines(), new_text.splitlines())

    added = []
    deleted = []
    
    for line in diff:
        if line.startswith('+ '):  # Line added
            added.append(line[2:])
        elif line.startswith('- '):  # Line deleted
            deleted.append(line[2:])
    
    # Print the added and deleted content
    if added:
        print("Added Text:")
        for line in added:
            print(f"  {line}")
    if deleted:
        print("Deleted Text:")
        for line in deleted:
            print(f"  {line}")

# File event handler for monitoring file changes
class FileChangeHandler(FileSystemEventHandler):
    def __init__(self, file_path):
        self.file_path = file_path
        self.last_content = read_text_file(file_path)  # Initial content of the file

    def on_modified(self, event):
        if event.src_path == self.file_path:  # Check if the modified file is the target one
            print(f"File '{self.file_path}' was modified.")
            
            # Read the current content of the file
            current_content = read_text_file(self.file_path)

            # Detect added and deleted text
            detect_changes(self.last_content, current_content)
            
            # Update the last_content with the new content for the next modification
            self.last_content = current_content

            # Reprocess the text to embed and store in ChromaDB
            embed_and_store_in_chromadb(current_content)

# Function to start the file monitoring
def start_file_monitor(file_path):
    event_handler = FileChangeHandler(file_path)
    observer = Observer()
    observer.schedule(event_handler, path=os.path.dirname(file_path), recursive=False)
    observer.start()

    try:
        while True:
            time.sleep(60)  # Check every 60 seconds (1 minute)
    except KeyboardInterrupt:
        observer.stop()
    observer.join()

# Example usage
if __name__ == "__main__":
    # Provide the actual path to your text document
    file_path = r"C:\Users\mohan\OneDrive\Desktop\GenAI\pdf\sample3.txt"  # Adjust the path accordingly
    
    # Start monitoring the file
    start_file_monitor(file_path)
