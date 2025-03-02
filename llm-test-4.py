import os
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"  # Disables GPU usage in TensorFlow

import tensorflow as tf
from transformers import TFGPT2LMHeadModel, GPT2Tokenizer, T5Tokenizer, TFAutoModelForSeq2SeqLM
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

def summarize_text(text, max_length=512):
    """Summarizes long text inputs before passing to GPT-2."""
    input_ids = summarizer_tokenizer("summarize: " + text, return_tensors="tf", truncation=True, max_length=1024).input_ids
    output_ids = summarizer_model.generate(input_ids, max_length=max_length)
    return summarizer_tokenizer.decode(output_ids.numpy()[0], skip_special_tokens=True)

def generate_response(prompt):
    """Generates a response using GPT-2 (TensorFlow version)."""
    print("Tokenizing input...")
    input_ids = tokenizer(prompt, return_tensors="tf").input_ids
    print("Tokenization complete")
    # Ensure input length does not exceed GPT-2's 1023 token limit
    if input_ids.shape[1] > 1023:
        input_ids = input_ids[:, -1023:]  # Keep only last 1023 tokens

    attention_mask = tf.ones_like(input_ids)

    print("Generating output ids...")
    with tf.device("/CPU:0"):
        output_ids = model.generate(
            input_ids,
            attention_mask=attention_mask,
            max_length=1023,  # Ensures generated sequence stays within limits
            # max_new_tokens=50,  # Limit output length
            pad_token_id=tokenizer.eos_token_id,  # Prevents index errors
            eos_token_id=tokenizer.eos_token_id  # Ensures early stopping
        )
    print("Output ids generated")

    return tokenizer.decode(output_ids.numpy()[0], skip_special_tokens=True)



# def generate_response(prompt):
#     """Generates a response using GPT-2 (TensorFlow version)."""
#     input_ids = tokenizer(prompt, return_tensors="tf").input_ids
    
#     # If input is too long, summarize before generating response
#     if input_ids.shape[1] > 1024:
#         prompt = summarize_text(prompt)
#         input_ids = tokenizer(prompt, return_tensors="tf").input_ids
    
#     attention_mask = tf.ones_like(input_ids)
#     output_ids = model.generate(
#         input_ids,
#         attention_mask=attention_mask,
#         max_new_tokens=200  # Generates up to 200 new tokens instead of limiting total length
#     )
#     return tokenizer.decode(output_ids.numpy()[0], skip_special_tokens=True)

def llm_pipeline(query):
    """Full pipeline with retrieval and response generation."""
    print("Retrieving context...")
    context = retrieve_context(query)
    print("Generating response...")
    augmented_prompt = "\n".join(context) + "\n\n" + query if context else query
    return generate_response(augmented_prompt)

if __name__ == "__main__":
    add_csv_to_vector_db("mini_data.csv", ["URL", "Content"])
    user_query = "What is SacHacks?"
    print("User query inputted")
    response = llm_pipeline(user_query)
    print("FULL RESPONSE:", response)
