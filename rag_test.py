import os
import faiss
import openai
import pickle
import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer
from dotenv import load_dotenv
import json
import gradio as gr

# Load environment variables
load_dotenv()
client = openai.OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))

# Constants
MODEL_NAME = "intfloat/e5-large-v2"
ITEM_DB_PATH = "item_db.csv"
FAISS_INDEX_PATH = "faiss_index.idx"
METADATA_PATH = "faiss_metadata.pkl"
TOP_K = 3
SIMILARITY_THRESHOLD = 0.4

# Load model + FAISS + metadata
model = SentenceTransformer(MODEL_NAME)
faiss_index = faiss.read_index(FAISS_INDEX_PATH)
with open(METADATA_PATH, "rb") as f:
    metadata = pickle.load(f)

def generate_step_back_query(original_query):
    prompt = f"""
You are an expert in reformulating user queries into more general questions that better capture the user's intent.

Your task is to generate a *step-back query*, which is a more abstract and generalized version of the original query. The step-back query should:
• Preserve the core meaning.
• Uncover implicit sub-questions.
• Rephrase specifics into broader concepts.
• Bridge the gap for retrieval systems.
• Use multi-sentence reformulation if helpful.

Original Query: "{original_query}"

Return only in the following JSON format:

{{ "step_back_query": "<your reformulated query>" }}
"""
    try:
        response = client.chat.completions.create(
            model="gpt-4o",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.3,
            max_tokens=200,
        )
        content = response.choices[0].message.content.strip()
        if content.startswith("```"):
            content = content.strip("`").strip()
            if content.lower().startswith("json"):
                content = content[4:].strip()
        parsed = json.loads(content)
        return parsed.get("step_back_query", "")
    except Exception as e:
        return f"Step-back query generation failed: {e}"

def get_avg_query_embedding(original_query, stepback_query):
    emb1 = model.encode(original_query, normalize_embeddings=True)
    emb2 = model.encode(stepback_query, normalize_embeddings=True)
    return (0.5 * emb1 + 0.5 * emb2).astype("float32")

def retrieve_top_k(embedding, top_k=TOP_K, threshold=SIMILARITY_THRESHOLD):
    scores, indices = faiss_index.search(embedding.reshape(1, -1), top_k)
    results = []
    for idx, score in zip(indices[0], scores[0]):
        if idx < len(metadata) and score <= threshold:
            results.append({**metadata[idx], "score": float(score)})
    return results

def generate_user_friendly_response(original_query, stepback_query, results):
    if not results:
        return "Sorry! I couldn’t find any good matches for your query. Try rephrasing it or being more specific."

    context = "\n\n".join([
        f"Item: {r['item_name']}\nDescription: {r['description']}" for r in results
    ])
    prompt = f"""
You are a helpful assistant. Based on the user's query and the following retrieved items, summarize them in a clear, friendly way that helps the user understand what might be relevant to them.

User query: "{original_query}"

Step-back reformulated query: "{stepback_query}"

Retrieved items:
{context}

Respond in a concise and easy-to-understand manner.
"""
    try:
        response = client.chat.completions.create(
            model="gpt-4o",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.3,
            max_tokens=300,
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        return f"Error generating GPT-4o response: {e}"

def rag_pipeline(user_query):
    stepback_query = generate_step_back_query(user_query)
    avg_embedding = get_avg_query_embedding(user_query, stepback_query)
    retrieved = retrieve_top_k(avg_embedding)
    final_response = generate_user_friendly_response(user_query, stepback_query, retrieved)
    return stepback_query, final_response

demo = gr.Interface(
    fn=rag_pipeline,
    inputs=gr.Textbox(label="Enter your query"),
    outputs=[
        gr.Textbox(label="Step-Back Query"),
        gr.Textbox(label="GPT-4o Response")
    ],
    title="RAG Assistant with Step-Back Prompting",
    description="Enter a query to see retrieval-augmented responses using E5 embeddings + GPT-4o"
)

demo.launch()