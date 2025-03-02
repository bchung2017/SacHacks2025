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

# Enable torch.compile() if available (for faster CPU execution)
if hasattr(torch, "compile"):
    model = torch.compile(model)

# Load SentenceTransformer for embedding retrieval
embed_model = SentenceTransformer("all-MiniLM-L6-v2", device=device)

# FAISS vector database setup
embedding_dim = 384  # Depends on embedding model
index = faiss.IndexFlatL2(embedding_dim)
documents = []  # Store (text, link, tags) tuples

def add_csv_to_vector_db(csv_file):
    """Loads text, tags, and links from a CSV file and adds them to the FAISS vector database."""
    global documents

    df = pd.read_csv(csv_file)
    required_columns = {"Text", "Tags", "Full Link"}
    if not required_columns.issubset(df.columns):
        raise ValueError(f"CSV file is missing required columns: {required_columns - set(df.columns)}")

    texts = df["Text"].astype(str).tolist()
    tags = df["Tags"].astype(str).tolist()
    links = df["Full Link"].tolist()

    combined_texts = [f"{t} | Tags: {tag}" for t, tag in zip(texts, tags)]  # Combine text and tags

    add_to_vector_db(combined_texts, links, tags)

def add_to_vector_db(texts, links, tags):
    """Adds valid text-link-tag triples to the FAISS vector database."""
    global documents

    valid_triples = [(text, link, tag) for text, link, tag in zip(texts, links, tags) if text and link]
    if not valid_triples:
        print("⚠️ No valid documents to add to FAISS.")
        return

    texts, links, tags = zip(*valid_triples)  # Unpack valid triples
    embeddings = embed_model.encode(texts, convert_to_numpy=True)
    index.add(embeddings)
    documents.extend(valid_triples)  # Store (text, link, tags) tuples


def retrieve_best_links(query, top_k=5):
    """Retrieves up to top_k best matching links for a query using FAISS and includes similarity scores."""
    if index.ntotal == 0:
        print("⚠️ FAISS index is empty.")
        return []

    query_embedding = embed_model.encode([query], convert_to_numpy=True)
    distances, indices = index.search(query_embedding, top_k)

    results = []
    for i, idx in enumerate(indices[0]):
        if idx >= len(documents):  # Ensure valid index
            continue
        
        best_text, best_link, best_tags = documents[idx]
        similarity_score = 1 / (1 + distances[0][i])  # Convert L2 distance to a similarity score (higher is better)

        results.append({"Text": best_text, "Link": best_link, "Tags": best_tags, "Similarity": similarity_score})

    return results


import re

def generate_response(prompt):
    """Generates a response using GPT-2 (optimized for CPU)."""
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
    """Retrieves best contexts and links, then generates a response."""
    print("Retrieving best links and contexts...")
    best_results = retrieve_best_links(query, top_k=5)

    if best_results:
        context = "\n".join([f"{r['Text']}\nTags: {r['Tags']}\nSimilarity: {r['Similarity']:.4f}" for r in best_results])
        prompt = f"Context:\n{context}\n\nQuestion: {query}\n\nAnswer:"
    else:
        prompt = f"Question: {query}\n\nAnswer:"

    print("Generating response...")
    response = generate_response(prompt)

    return response, best_results


if __name__ == "__main__":
    # Load CSV into FAISS (Ensure the CSV file exists)
    csv_file = "mini_codecrafters_data_2.csv"  # Change this to your actual CSV file
    try:
        add_csv_to_vector_db(csv_file)
        print("CSV data added to FAISS successfully.")
    except Exception as e:
        print(f"Error loading CSV: {e}")

    # Query processing
    user_query = "I want to learn about builtins from scratch for future PHP development"
    response, best_results = llm_pipeline(user_query)

    print(f"\nResponse: {response}")
    print("\nRecommended Links:")
    for res in best_results:
        print(f" - {res['Link']} (Similarity: {res['Similarity']:.4f})")
