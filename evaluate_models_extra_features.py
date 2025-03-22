import pandas as pd
import time

from model_registry import ModelRegistry
from item_db_search_engine_updated import VectorSearchEngine 

def load_queries(filepath, sample_size=1000, random_state=42):
    df = pd.read_csv(filepath)
    sample_df = df.sample(n=sample_size, random_state=random_state)
    print(f"{len(sample_df)} queries loaded from {filepath}")
    return sample_df['query'].tolist()

def evaluate_model(model_name, queries, vector_search_engine, min_sim_threshold=0.2, max_num_results=5):
    total_top_similarity = 0.0
    total_weighted_similarity = 0.0
    total_response_time = 0.0
    coverage_count = 0
    total_results_count = 0

    base_weights = [0.5, 0.2, 0.15, 0.1, 0.05]
    detailed_results = []

    print(f"Evaluating model: {model_name}")
    for idx, query in enumerate(queries):
        start_time = time.time()
        results = vector_search_engine.search(query, model_name, max_num_results, min_sim_threshold)
        response_time = time.time() - start_time

        total_response_time += response_time

        if not results.empty:
            top_similarity = results.iloc[0]['similarity_score']
            total_top_similarity += top_similarity
            if top_similarity >= min_sim_threshold:
                coverage_count += 1
            total_results_count += len(results)
            scores = results['similarity_score'].tolist()[:max_num_results]
            weights = base_weights[:len(scores)]
            weight_sum = sum(weights)
            normalized_weights = [w / weight_sum for w in weights]
            weighted_score = sum(s * w for s, w in zip(scores, normalized_weights))
        else:
            top_similarity = 0
            weighted_score = 0

        total_weighted_similarity += weighted_score

        detailed_results.append({
            'query': query,
            'top_similarity': top_similarity,
            'weighted_top5_similarity': weighted_score,
            'num_results': len(results),
            'response_time': response_time
        })

        if (idx + 1) % 100 == 0 or (idx + 1) == len(queries):
            print(f"Model {model_name} - {idx + 1}/{len(queries)} processed. Top-1: {top_similarity:.4f}, Weighted Top-5: {weighted_score:.4f}, Results: {len(results)}, Time: {response_time:.4f}s")

    metrics = {
        'model_name': model_name,
        'avg_top_similarity': total_top_similarity / len(queries),
        'avg_weighted_top5_similarity': total_weighted_similarity / len(queries),
        'coverage_percent': (coverage_count / len(queries)) * 100,
        'avg_num_results': total_results_count / len(queries),
        'avg_response_time': total_response_time / len(queries)
    }

    return metrics, detailed_results

def main():
    item_db_path = 'item_db_enriched.csv'  # NEW enriched item DB
    item_searches_path = 'item_searches.csv'

    queries = load_queries(item_searches_path, sample_size=1000, random_state=42)
    vector_search_engine = VectorSearchEngine(item_db_path)
    registry = ModelRegistry()
    available_models = registry.available_models()

    all_metrics = []
    all_detailed_results = {}

    for model_name in available_models:
        print(f"\n--- Evaluating {model_name} ---")
        metrics, detailed_results = evaluate_model(model_name, queries, vector_search_engine)
        all_metrics.append(metrics)
        all_detailed_results[model_name] = detailed_results

    pd.DataFrame(all_metrics).to_csv('evaluation_metrics_enriched.csv', index=False)
    print("Saved metrics to evaluation_metrics_enriched.csv")

    with open("evaluation_summary_enriched.txt", "w") as f:
        f.write("------Model Evaluation Summary Report (Enriched DB)------\n------------\n")
        for m in all_metrics:
            f.write(f"Model: {m['model_name']}\n")
            f.write(f"  Average Top-1 Similarity: {m['avg_top_similarity']:.4f}\n")
            f.write(f"  Average Weighted Top-5 Similarity: {m['avg_weighted_top5_similarity']:.4f}\n")
            f.write(f"  Query Coverage (%): {m['coverage_percent']:.2f}\n")
            f.write(f"  Average Number of Results: {m['avg_num_results']:.2f}\n")
            f.write(f"  Average Response Time (s): {m['avg_response_time']:.4f}\n")
            f.write("------------\n")

    print("Saved summary to evaluation_summary_enriched.txt")

if __name__ == "__main__":
    main()