from flask import Flask, request, jsonify
from flask_cors import CORS
import requests
from bs4 import BeautifulSoup
import pandas as pd
import nltk
from nltk.corpus import stopwords
import re
import os
import torch
import faiss
import numpy as np
import pandas as pd
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer
from sentence_transformers import SentenceTransformer
import re
from flask import Flask, request, jsonify
from flask_cors import CORS

# Force CPU usage
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"  # Disables GPU
device = "cpu"
print(f"Using device: {device}")

nltk.download("stopwords")

app = Flask(__name__)
CORS(app)

@app.route("/scrape", methods=["POST"])
def scrape():
    data = request.json
    url = data.get("url")
    url = url + "/tracks/php"

    if not url:
        return jsonify({"error": "URL is required"}), 400

    try:
        headers = {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/110.0.0.0 Safari/537.36"
        }
        response = requests.get(url, headers=headers)

        if response.status_code != 200:
            return jsonify({"error": f"Failed to retrieve the page (Status: {response.status_code})"}), 400

        soup = BeautifulSoup(response.text, "html.parser")
        stop_words = set(stopwords.words("english"))
        data = []
        print("Scraping URL...")
        
        for section in soup.find_all("div", class_="bg-white dark:bg-gray-850 rounded-md shadow-sm border border-gray-200 dark:border-white/5 relative w-full group mb-4"):
            section_title_div = section.find("div", class_="text-xl font-semibold text-gray-800 dark:text-gray-200")
            section_title = section_title_div.get_text(strip=True) if section_title_div else ""

            for link in section.find_all("a", class_="ember-view block hover:bg-gray-50 dark:hover:bg-gray-700/50 py-1.5 -mx-1.5 px-1.5 rounded"):
                href = link.get("href")
                text_div = link.find("div", class_="prose dark:prose-invert prose-sm")

                if href and text_div:
                    full_link = f"{url}{href}"
                    text = text_div.get_text(strip=True)

                    combined_text = f"{section_title} {text}"
                    words = set(re.findall(r'\b\w+\b', combined_text.lower())) - stop_words
                    tags = " | ".join(sorted(words))

                    data.append({"Full Link": full_link, "Text": text, "Tags": tags})

        # Save data to CSV
        df = pd.DataFrame(data)
        df.to_csv("scraped_data.csv", index=False)
        print("Data saved to scraped_data.csv")

        return jsonify(data)

    except Exception as e:
        return jsonify({"error": str(e)}), 500


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
    """Retrieves up to top_k best matching links for a query using FAISS and rescales similarity scores."""
    if index.ntotal == 0:
        print("⚠️ FAISS index is empty.")
        return []

    query_embedding = embed_model.encode([query], convert_to_numpy=True)
    distances, indices = index.search(query_embedding, top_k)

    results = []
    similarities = []

    # Extract similarity scores from distances
    for i, idx in enumerate(indices[0]):
        if idx >= len(documents):  # Ensure valid index
            continue
        
        best_text, best_link, best_tags = documents[idx]
        similarity_score = float(1 / (1 + distances[0][i]))  # Convert L2 distance to similarity
        similarities.append(similarity_score)

        results.append({
            "Link": best_link,
            "Tags": best_tags,
            "Similarity": similarity_score  # Temporary, will be rescaled below
        })

    # Ensure we have results to normalize
    if not results:
        return []

    # Rescale similarities so the highest score is always >= 90%
    max_similarity = max(similarities)  # Get the highest similarity score
    scaling_factor = 0.99 / max_similarity  # Scale the highest similarity to 90%

    for result in results:
        result["Similarity"] = min(1.0, result["Similarity"] * scaling_factor)  # Rescale & cap at 100%

    return results



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

    return response.strip()

def llm_pipeline(query):
    """Retrieves best contexts and links, then generates a response."""
    print("Retrieving best links and contexts...")
    best_results = retrieve_best_links(query, top_k=5)

    # if best_results:
    #     context = "\n".join([f"{r['Text']}\nTags: {r['Tags']}\nSimilarity: {r['Similarity']:.4f}" for r in best_results])
    #     prompt = f"Context:\n{context}\n\nQuestion: {query}\n\nAnswer:"
    # else:
    #     prompt = f"Question: {query}\n\nAnswer:"
    print("\nRecommended Links:")
    for res in best_results:
        print(f" - {res['Link']} (Similarity: {res['Similarity']:.4f}) | Tags: {res['Tags']}")

    # print("Generating response...")
    # response = generate_response(prompt)

    # return response, best_results

    return best_results

@app.route("/search", methods=["POST"])
def search():
    data = request.json
    query = data.get("query", "")

    if not query:
        return jsonify({"error": "Query is required"}), 400

    best_results = llm_pipeline(query)

    return jsonify(best_results)


if __name__ == "__main__":
    csv_file = "extracted_codecrafters_links.csv"  # Change this to your actual CSV file
    try:
        add_csv_to_vector_db(csv_file)
        print("CSV data added to FAISS successfully.")
    except Exception as e:
        print(f"Error loading CSV: {e}")
    
    app.run(debug=True)
