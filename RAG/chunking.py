import spacy
import nltk
import re
from transformers import pipeline
import fitz  # PyMuPDF
import json
from transformers import BertTokenizer

# Load spaCy model for syntactic and entity chunking
nlp = spacy.load("en_core_web_sm")

# Function to read PDF
def read_pdf(file_path):
    document = fitz.open(r"C:\Users\mohan\OneDrive\Desktop\GenAI\pdf\Most-Common-F1-Visa-Interview-Questions-Asked-Recently.pdf")
    text = ""
    for page_num in range(document.page_count):
        page = document.load_page(page_num)
        text += page.get_text("text")
    return text

# Function for syntactic chunking
def syntactic_chunking(text):
    doc = nlp(text)
    chunks = []
    for np in doc.noun_chunks:
        chunks.append(np.text)
    return chunks

# Initialize the BERT tokenizer
tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")

# Function for semantic chunking (without embedding) - splits text into chunks of 512 tokens
def semantic_chunking(text, max_length=512):
    # Tokenize the entire text
    tokens = tokenizer.encode(text, truncation=False, padding=False)
    
    # Create chunks of max_length (up to 512 tokens per chunk)
    chunks = [tokens[i:i + max_length] for i in range(0, len(tokens), max_length)]
    
    # Decode each chunk back to text
    chunk_texts = [tokenizer.decode(chunk, skip_special_tokens=True) for chunk in chunks]
    
    return chunk_texts

# Function for fixed-length chunking
def fixed_length_chunking(text, length=512):
    words = text.split()
    chunks = [words[i:i + length] for i in range(0, len(words), length)]
    return [' '.join(chunk) for chunk in chunks]

# Function for sliding window chunking
def sliding_window_chunking(text, window_size=512, overlap=50):
    words = text.split()
    chunks = []
    for i in range(0, len(words), window_size - overlap):
        chunk = words[i:i + window_size]
        chunks.append(' '.join(chunk))
    return chunks

# Function for paragraph-level chunking
def paragraph_level_chunking(text):
    paragraphs = text.split("\n\n")
    return [para.strip() for para in paragraphs if para.strip()]

# Function for sentence-level chunking
def sentence_level_chunking(text):
    sentences = nltk.sent_tokenize(text)
    return sentences

# Function for entity-based chunking
def entity_based_chunking(text):
    doc = nlp(text)
    entities = []
    for ent in doc.ents:
        entities.append((ent.text, ent.label_))
    return entities

# Main chunking function that stores all outputs
def chunk_pdf(file_path):
    text = read_pdf(file_path)
    
    # Apply all chunking methods
    syntactic_chunks = syntactic_chunking(text)
    semantic_chunks = semantic_chunking(text)
    fixed_length_chunks = fixed_length_chunking(text)
    sliding_window_chunks = sliding_window_chunking(text)
    paragraph_chunks = paragraph_level_chunking(text)
    sentence_chunks = sentence_level_chunking(text)
    entity_chunks = entity_based_chunking(text)
    
    # Save outputs as JSON (or any other format you prefer)
    outputs = {
        "syntactic_chunks": syntactic_chunks,
        "semantic_chunks": semantic_chunks,
        "fixed_length_chunks": fixed_length_chunks,
        "sliding_window_chunks": sliding_window_chunks,
        "paragraph_chunks": paragraph_chunks,
        "sentence_chunks": sentence_chunks,
        "entity_chunks": entity_chunks
    }
    
    # Save outputs to a JSON file
    with open("chunking_output.json", "w") as f:
        json.dump(outputs, f, indent=4)

    return outputs
