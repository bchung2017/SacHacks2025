import os
import torch
import faiss
import numpy as np
import pandas as pd
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer
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
documents = []  # Store (text, link) pairs


def add_csv_to_vector_db(csv_file, text_columns):
    """Loads text data from a CSV file and adds it to the FAISS vector database."""
    global documents

    df = pd.read_csv(csv_file)
    if not all(col in df.columns for col in text_columns + ["Full Link"]):
        raise ValueError(f"Some required columns {text_columns + ['Full Link']} not found in CSV.")

    texts = df[text_columns].astype(str).agg(" ".join, axis=1).tolist()
    links = df["Full Link"].tolist()

    add_to_vector_db(texts, links)


def add_to_vector_db(texts, links):
    """Adds valid text-link pairs to the FAISS vector database."""
    global documents

    valid_pairs = [(text, link) for text, link in zip(texts, links) if text and link]
    if not valid_pairs:
        print("⚠️ No valid documents to add to FAISS.")
        return

    texts, links = zip(*valid_pairs)  # Unpack valid pairs
    embeddings = embed_model.encode(texts, convert_to_numpy=True)
    index.add(embeddings)
    documents.extend(valid_pairs)  # Store (text, link) pairs


def retrieve_best_link(query, top_k=1):
    """Retrieves the best matching text and link for a query using FAISS, always returning at least one result."""
    if index.ntotal == 0:
        print("⚠️ FAISS index is empty.")
        return None, None

    query_embedding = embed_model.encode([query], convert_to_numpy=True)
    distances, indices = index.search(query_embedding, top_k)

    best_index = indices[0][0]  # Always take the top result

    if best_index >= len(documents):  # Ensure valid index
        print("⚠️ FAISS returned an out-of-range index.")
        return None, None

    best_text, best_link = documents[best_index]  # Extract best text-link pair
    return best_text, best_link


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


def llm_pipeline(query):
    """Retrieves best context and link, then generates a response."""
    print("Retrieving best link and context...")
    best_text, best_link = retrieve_best_link(query)

    if best_text:
        prompt = f"Context:\n{best_text}\n\nQuestion: {query}\n\nAnswer:"
    else:
        prompt = f"Question: {query}\n\nAnswer:"

    print("Generating response...")
    response = generate_response(prompt)

    return response, best_link


if __name__ == "__main__":
    # Load CSV into FAISS (Ensure the CSV file exists)
    csv_file = "mini_codecrafters_data.csv"  # Change this to your actual CSV file
    try:
        add_csv_to_vector_db(csv_file, ["Text"])
        print("CSV data added to FAISS successfully.")
    except Exception as e:
        print(f"Error loading CSV: {e}")

    # Query processing
    user_query = "I want to learn about builtins from scratch for future PHP development"
    response, best_link = llm_pipeline(user_query)

    print(f"Response: {response}")
    print(f"Recommended Link: {best_link if best_link else 'No link found'}")
