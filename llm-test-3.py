import os
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"  # Disables GPU usage in TensorFlow

import tensorflow as tf
from transformers import TFGPT2LMHeadModel, GPT2Tokenizer
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np
import pandas as pd


# Load model and tokenizer
model_name = "gpt2"
tokenizer = GPT2Tokenizer.from_pretrained(model_name)
model = TFGPT2LMHeadModel.from_pretrained(model_name)

embed_model = SentenceTransformer("all-MiniLM-L6-v2")

# Placeholder FAISS vector database
embedding_dim = 384  # Depends on embedding model
index = faiss.IndexFlatL2(embedding_dim)
documents = []

def add_csv_to_vector_db(csv_file, text_columns):
    """Loads text data from a CSV file with multiple text columns and adds it to the vector database."""
    df = pd.read_csv(csv_file)
    missing_columns = [col for col in text_columns if col not in df.columns]
    if missing_columns:
        raise ValueError(f"Columns {missing_columns} not found in CSV file.")
    texts = df[text_columns].astype(str).agg(" ".join, axis=1).dropna().tolist()
    add_to_vector_db(texts)

def add_to_vector_db(texts):
    """Adds documents to the vector database."""
    global documents
    embeddings = embed_model.encode(texts, convert_to_numpy=True)
    index.add(embeddings)
    documents.extend(texts)

def retrieve_context(query, top_k=3):
    """Retrieves relevant documents from the vector DB."""
    if index.ntotal == 0:
        return []
    query_embedding = embed_model.encode([query], convert_to_numpy=True)
    distances, indices = index.search(query_embedding, top_k)
    return [documents[i] for i in indices[0] if i < len(documents)]

def generate_response(prompt):
    """Generates a response using Mistral (TensorFlow version)."""
    input_ids = tokenizer(prompt, return_tensors="tf")
    input_ids = input_ids.input_ids
    attention_mask = tf.ones_like(input_ids)
    output_ids = model.generate(
        input_ids,
        attention_mask=attention_mask,
        max_new_tokens=200  # Generates up to 200 new tokens instead of limiting total length
    )

    return tokenizer.decode(output_ids.numpy()[0], skip_special_tokens=True)

def llm_pipeline(query):
    """Full pipeline with retrieval and response generation."""
    context = retrieve_context(query)
    augmented_prompt = "\n".join(context) + "\n\n" + query if context else query
    return generate_response(augmented_prompt)

if __name__ == "__main__":
    # add_to_vector_db(["Mistral is a powerful language model.", "FAISS is used for vector search."])
    add_csv_to_vector_db("sachacks_scraped_data_selenium_1.csv", ["URL", "Content"])
    user_query = "What is used for vector search?"
    response = llm_pipeline(user_query)
    print("Response:", response)