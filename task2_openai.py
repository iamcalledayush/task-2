import pandas as pd
import numpy as np
import time
import openai
from tqdm import tqdm
import os
from dotenv import load_dotenv

load_dotenv()

client = openai.OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))
EMBEDDING_MODEL = "text-embedding-ada-002" 

def get_embedding(text):
    try:
        response = openai.embeddings.create(
            input=[text],
            model=EMBEDDING_MODEL
        )
        return response.data[0].embedding
    except Exception as e:
        print(f"Embedding failed for: {text} - Error: {e}")
        return [0.0] * 1536  

def cosine_similarity(a, b):
    a = np.array(a)
    b = np.array(b)
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))


item_df = pd.read_csv("item_db_enriched.csv")
item_df["combined_text"] = item_df["item_name"] + " - " + item_df["description"]
item_embeddings = []

print("Embedding items...")
for text in tqdm(item_df["combined_text"].tolist()):
    item_embeddings.append(get_embedding(text))


query_df = pd.read_csv("item_searches.csv").sample(n=1000, random_state=42)
queries = query_df["query"].tolist()


results = []
print("Evaluating queries...")
for i, query in enumerate(tqdm(queries)):
    start_time = time.time()
    query_emb = get_embedding(query)

    sims = [cosine_similarity(query_emb, item_emb) for item_emb in item_embeddings]
    item_df["similarity_score"] = sims
    top_k = item_df.sort_values(by="similarity_score", ascending=False).head(5)

    top1 = top_k.iloc[0]["similarity_score"]
    weighted = np.average(top_k["similarity_score"], weights=[0.5, 0.2, 0.15, 0.1, 0.05])
    count = sum(score >= 0.2 for score in top_k["similarity_score"])
    duration = time.time() - start_time

    results.append({
        "query": query,
        "top_similarity": top1,
        "weighted_top5_similarity": weighted,
        "num_results": len(top_k),
        "response_time": duration
    })

    if (i+1) % 100 == 0:
        print(f"Processed {i+1}/1000")


df = pd.DataFrame(results)
metrics = {
    "model": EMBEDDING_MODEL,
    "avg_top_similarity": df["top_similarity"].mean(),
    "avg_weighted_top5_similarity": df["weighted_top5_similarity"].mean(),
    "coverage_percent": 100.0 * (df["top_similarity"] >= 0.2).sum() / len(df),
    "avg_num_results": df["num_results"].mean(),
    "avg_response_time": df["response_time"].mean()
}

print("\n--- Evaluation Metrics ---")
for k, v in metrics.items():
    print(f"{k}: {v:.4f}" if isinstance(v, float) else f"{k}: {v}")


df.to_csv("openai_eval_detailed_results.csv", index=False)
pd.DataFrame([metrics]).to_csv("openai_eval_summary.csv", index=False)