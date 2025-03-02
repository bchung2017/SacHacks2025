import os
import torch
import faiss
import numpy as np
import pandas as pd
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer, AutoModelForSeq2SeqLM
from sentence_transformers import SentenceTransformer

# Force CPU usage
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"  # Disables GPU
device = "cpu"
print(f"Using device: {device}")

# Load GPT-2 model and tokenizer
model_name = "gpt2"
tokenizer = AutoTokenizer.from_pretrained(model_name)

# Load GPT-2 on CPU (No quantization, but optimized)
model = AutoModelForCausalLM.from_pretrained(model_name).to(device).eval()

# Enable `torch.compile()` if available (for faster CPU execution)
if hasattr(torch, "compile"):
    model = torch.compile(model)

# Load SentenceTransformer for embedding retrieval
embed_model = SentenceTransformer("all-MiniLM-L6-v2", device=device)

# FAISS vector database setup
embedding_dim = 384  # Depends on embedding model
index = faiss.IndexFlatL2(embedding_dim)
documents = []

# Load T5 summarization model (if needed)
summarizer_model_name = "t5-small"
summarizer_tokenizer = AutoTokenizer.from_pretrained(summarizer_model_name)
summarizer_model = AutoModelForSeq2SeqLM.from_pretrained(summarizer_model_name).to(device).eval()

def add_csv_to_vector_db(csv_file, text_columns):
    """Loads text data from a CSV file and adds it to the FAISS vector database."""
    df = pd.read_csv(csv_file)
    missing_columns = [col for col in text_columns if col not in df.columns]
    if missing_columns:
        raise ValueError(f"Columns {missing_columns} not found in CSV file.")
    
    texts = df[text_columns].astype(str).agg(" ".join, axis=1).dropna().tolist()
    add_to_vector_db(texts)

def add_to_vector_db(texts):
    """Adds documents to the FAISS vector database."""
    global documents
    embeddings = embed_model.encode(texts, convert_to_numpy=True)
    index.add(embeddings)
    documents.extend(texts)

def retrieve_context(query, top_k=3, max_chars=500):
    """Retrieves relevant documents from the FAISS vector database and trims long context."""
    if index.ntotal == 0:
        return []
    
    query_embedding = embed_model.encode([query], convert_to_numpy=True)
    distances, indices = index.search(query_embedding, top_k)
    
    # Get unique results while limiting length
    retrieved_docs = set()
    total_chars = 0
    for i in indices[0]:
        if i < len(documents):
            doc = documents[i]
            if doc not in retrieved_docs and total_chars + len(doc) <= max_chars:
                retrieved_docs.add(doc)
                total_chars += len(doc)

    return list(retrieved_docs)


def summarize_text(text, max_length=512):
    """Summarizes long text inputs before passing to GPT-2."""
    input_ids = summarizer_tokenizer("summarize: " + text, return_tensors="pt", truncation=True, max_length=1024).input_ids.to(device)
    
    with torch.no_grad():
        output_ids = summarizer_model.generate(input_ids, max_length=max_length)
    
    return summarizer_tokenizer.decode(output_ids[0], skip_special_tokens=True)

import re

def generate_response(prompt):
    """Generates a response using GPT-2 (optimized for CPU) with tqdm progress tracking."""
    print("Tokenizing input...")
    input_ids = tokenizer(prompt, return_tensors="pt").input_ids.to(device)
    print("Tokenization complete")

    # Ensure input length does not exceed GPT-2's token limit
    if input_ids.shape[1] > 1023:
        input_ids = input_ids[:, -1023:]

    attention_mask = torch.ones_like(input_ids)

    print("Generating output ids...")

    with torch.no_grad():
        output_ids = model.generate(
            input_ids,
            attention_mask=attention_mask,
            max_new_tokens=50,
            pad_token_id=tokenizer.eos_token_id,
            eos_token_id=tokenizer.eos_token_id
        )

    print("Output ids generated")

    # Decode output
    response = tokenizer.decode(output_ids[0], skip_special_tokens=True)

    # Extract only the answer part (removes echoed prompt)
    if "Answer:" in response:
        response = response.split("Answer:")[-1].strip()

    # Truncate at the last complete sentence
    response = re.split(r'(?<=\.)\s', response)  # Split at sentence boundaries
    if response:
        response = " ".join(response[:-1]) + "."  # Remove any unfinished part

    # Ensure only ONE period at the end
    response = re.sub(r'\.+$', '.', response)  # Remove extra trailing periods

    return response.strip()




    print("Output ids generated")
    return tokenizer.decode(output_ids[0], skip_special_tokens=True)

def llm_pipeline(query):
    """Full pipeline with retrieval and response generation."""
    print("Retrieving context...")
    context = retrieve_context(query)

    # Format prompt properly to avoid context repetition
    if context:
        context_str = "\n".join(context).strip()
        context_str = context_str[:500]  # Limit context to 500 chars

        prompt = (
            f"Context:\n{context_str}\n\n"
            f"Question: {query}\n\n"
            f"Answer:"
        )
    else:
        prompt = f"Question: {query}\n\nAnswer:"

    print("Generating response...")
    return generate_response(prompt)



if __name__ == "__main__":
    add_csv_to_vector_db("mini_codecrafters_data.csv", ["Full Link", "Text"])
    user_query = "I want to learn to build a shell?"
    print("User query inputted")
    response = llm_pipeline(user_query)
    print("FULL RESPONSE:", response)
