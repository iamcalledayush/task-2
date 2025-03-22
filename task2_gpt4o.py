import pandas as pd
import time
import os
import openai
from concurrent.futures import ThreadPoolExecutor, as_completed


from model_registry import ModelRegistry
from item_db_search_engine import VectorSearchEngine

import os
from dotenv import load_dotenv

load_dotenv()

# GPT-4o API 
client = openai.OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))

paraphrased_path = "paraphrased_queries_gpt4o.csv"

# Paraphrasing function
def transform_query_gpt(query: str) -> str:
    try:
        response = client.chat.completions.create(
            model="gpt-4o",
            messages=[
                {"role": "system", "content": "You are a helpful assistant that improves queries for search. You must keep the original meaning intact and only fix grammar or clarity if needed. Do not change any keywords, reword with synonyms, or add extra details."},
                {"role": "user", "content": f"Improve the grammar of this search query without changing its meaning: {query}"}
            ],
            temperature=0.7,
            max_tokens=256,
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        print(f"Error paraphrasing query: {query}\nError: {e}")
        return query  # Fallback to original

def load_queries(filepath, sample_size=1000, random_state=42):
    df = pd.read_csv(filepath)
    sample_df = df.sample(n=sample_size, random_state=random_state)
    print(f"Loaded {len(sample_df)} queries from {filepath}")
    return sample_df['query'].tolist()

def cache_paraphrased_queries(queries):
    paraphrased = []
    if os.path.exists(paraphrased_path):
        print("Loading cached paraphrased queries...")
        df = pd.read_csv(paraphrased_path)
        return dict(zip(df['original_query'], df['paraphrased_query']))

    print("Generating and caching paraphrased queries using GPT-4o...")
    with ThreadPoolExecutor(max_workers=5) as executor:
        future_to_query = {executor.submit(transform_query_gpt, q): q for q in queries}
        for future in as_completed(future_to_query):
            query = future_to_query[future]
            try:
                result = future.result()
            except Exception as e:
                result = query
            paraphrased.append((query, result))

    df = pd.DataFrame(paraphrased, columns=["original_query", "paraphrased_query"])
    df.to_csv(paraphrased_path, index=False)
    print(f"Saved paraphrased queries to {paraphrased_path}")
    return dict(paraphrased)

def evaluate_model(model_name, queries, paraphrased_map, vector_search_engine, min_sim_threshold=0.2, max_num_results=5):
    total_top_similarity = 0.0
    total_weighted_similarity = 0.0
    total_response_time = 0.0
    coverage_count = 0
    total_results_count = 0
    num_queries = len(queries)
    detailed_results = []
    base_weights = [0.5, 0.2, 0.15, 0.1, 0.05]

    print(f"Starting evaluation for model: {model_name}")

    for idx, query in enumerate(queries):
        transformed_query = paraphrased_map.get(query, query)
        start_time = time.time()
        results = vector_search_engine.search(transformed_query, model_name, max_num_results=max_num_results, min_sim_threshold=min_sim_threshold)
        response_time = time.time() - start_time
        total_response_time += response_time

        if not results.empty:
            top_similarity = results.iloc[0]['similarity_score']
            total_top_similarity += top_similarity
            if top_similarity >= min_sim_threshold:
                coverage_count += 1
            total_results_count += len(results)
        else:
            top_similarity = 0

        scores = results['similarity_score'].tolist()[:max_num_results] if not results.empty else []
        selected_weights = base_weights[:len(scores)]
        weight_sum = sum(selected_weights)
        normalized_weights = [w / weight_sum for w in selected_weights] if weight_sum > 0 else []
        weighted_score = sum(s * w for s, w in zip(scores, normalized_weights)) if scores else 0

        total_weighted_similarity += weighted_score

        detailed_results.append({
            'original_query': query,
            'transformed_query': transformed_query,
            'top_similarity': top_similarity,
            'weighted_top5_similarity': weighted_score,
            'num_results': len(results),
            'response_time': response_time
        })

        if (idx + 1) % 100 == 0 or (idx + 1) == num_queries:
            print(f"Model {model_name} - Processed {idx + 1}/{num_queries} queries. Top-1: {top_similarity:.4f}, Weighted Top-5: {weighted_score:.4f}, Results: {len(results)}, Time: {response_time:.4f}s")

    metrics = {
        'model_name': model_name,
        'avg_top_similarity': total_top_similarity / num_queries,
        'avg_weighted_top5_similarity': total_weighted_similarity / num_queries,
        'coverage_percent': (coverage_count / num_queries) * 100,
        'avg_num_results': total_results_count / num_queries,
        'avg_response_time': total_response_time / num_queries
    }

    print(f"Completed evaluation for model: {model_name}\nMetrics: {metrics}\n")
    return metrics, detailed_results

def main():
    item_db_path = 'item_db.csv'
    item_searches_path = 'item_searches.csv'
    queries = load_queries(item_searches_path)
    paraphrased_map = cache_paraphrased_queries(queries)
    vector_search_engine = VectorSearchEngine(item_db_path)
    registry = ModelRegistry()
    all_metrics, all_detailed_results = [], {}

    for model_name in registry.available_models():
        metrics, detailed = evaluate_model(
            model_name,
            queries,
            paraphrased_map,
            vector_search_engine
        )
        all_metrics.append(metrics)
        all_detailed_results[model_name] = detailed

    pd.DataFrame(all_metrics).to_csv('evaluation_metrics_transformed.csv', index=False)
    with open("evaluation_summary_transformed.txt", "w") as f:
        f.write("------Model Evaluation Summary Report (Query Transformed with GPT-4o)------\n------------\n")
        for m in all_metrics:
            f.write(f"Model: {m['model_name']}\n")
            f.write(f"  Average Top-1 Similarity: {m['avg_top_similarity']:.4f}\n")
            f.write(f"  Average Weighted Top-5 Similarity: {m['avg_weighted_top5_similarity']:.4f}\n")
            f.write(f"  Query Coverage (%): {m['coverage_percent']:.2f}\n")
            f.write(f"  Average Number of Results: {m['avg_num_results']:.2f}\n")
            f.write(f"  Average Response Time (s): {m['avg_response_time']:.4f}\n------------\n")

if __name__ == "__main__":
    main()
