import os
import faiss
import openai
import pandas as pd
import pickle
import numpy as np
from tqdm import tqdm
from sentence_transformers import SentenceTransformer
from dotenv import load_dotenv

# Load environment variables
load_dotenv()
client = openai.OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))

# Constants
MODEL_NAME = "intfloat/e5-large-v2"
ITEM_DB_PATH = "item_db.csv"
QUERY_CSV_PATH = "stepback_enhanced_queries_sampled.csv"
FAISS_INDEX_PATH = "faiss_index.idx"
METADATA_PATH = "faiss_metadata.pkl"
OUTPUT_PATH = "rag_results_sample.csv"

# Load model
model = SentenceTransformer(MODEL_NAME)

# Step 1: Create and save FAISS index if not exists
def create_faiss_index():
    if os.path.exists(FAISS_INDEX_PATH) and os.path.exists(METADATA_PATH):
        print("FAISS index and metadata already exist. Skipping creation.")
        return

    print("Creating FAISS index...")
    df = pd.read_csv(ITEM_DB_PATH)
    name_embeddings = model.encode(df["item_name"].tolist(), normalize_embeddings=True)
    desc_embeddings = model.encode(df["description"].tolist(), normalize_embeddings=True)
    combined_embeddings = 0.5 * name_embeddings + 0.5 * desc_embeddings

    index = faiss.IndexFlatL2(combined_embeddings.shape[1])
    index.add(combined_embeddings)
    faiss.write_index(index, FAISS_INDEX_PATH)

    with open(METADATA_PATH, "wb") as f:
        pickle.dump(df.to_dict(orient="records"), f)

    print("FAISS index and metadata saved.")

# Step 2: Retrieve top-k items from FAISS
def retrieve_top_k_items(embedding, top_k=3, threshold=0.4):
    index = faiss.read_index(FAISS_INDEX_PATH)
    with open(METADATA_PATH, "rb") as f:
        metadata = pickle.load(f)

    scores, indices = index.search(np.array([embedding]), top_k)
    results = []
    for idx, score in zip(indices[0], scores[0]):
        if idx < len(metadata) and score <= threshold:
            item = metadata[idx]
            item["score"] = float(score)
            results.append(item)
    return results

# Step 3: Ask GPT-4o to generate a human-friendly response
def ask_gpt4o(query, results):
    context = "\n\n".join([
        f"Item: {r['item_name']}\nDescription: {r['description']}" for r in results
    ])
    prompt = f"""
You are a helpful assistant. Based on the user's query and the following retrieved items, summarize them in a clear, friendly way that helps the user understand what might be relevant to them.

User query: "{query}"

Retrieved items:
{context}

Respond in a concise and easy-to-understand manner.
"""
    try:
        response = client.chat.completions.create(
            model="gpt-4o",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.3,
            max_tokens=300
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        print(f"Error during GPT-4o response: {e}")
        return "Error in response generation."

# Step 4: Main RAG pipeline for 10 queries
def main():
    create_faiss_index()
    queries_df = pd.read_csv(QUERY_CSV_PATH).head(10)  # Only first 10 queries

    results = []
    for _, row in tqdm(queries_df.iterrows(), total=len(queries_df), desc="Running RAG pipeline"):
        original_query = row["original_query"]
        enhanced_query = row["enhanced_query"]
        emb1 = model.encode(original_query, normalize_embeddings=True)
        emb2 = model.encode(enhanced_query, normalize_embeddings=True)
        avg_embedding = 0.5 * emb1 + 0.5 * emb2

        top_items = retrieve_top_k_items(avg_embedding)
        gpt_response = ask_gpt4o(original_query, top_items)

        results.append({
            "original_query": original_query,
            "enhanced_query": enhanced_query,
            "gpt4o_response": gpt_response
        })

    pd.DataFrame(results).to_csv(OUTPUT_PATH, index=False)
    print(f"Saved responses to {OUTPUT_PATH}")

if __name__ == "__main__":
    main()
