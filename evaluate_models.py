import pandas as pd
import time
import os


from model_registry import ModelRegistry 
from item_db_search_engine import VectorSearchEngine 

def load_queries(filepath, sample_size=1000, random_state=42):
    """
    Load queries from item_searches.csv and randomly sample a subset for evaluation.
    """
    df = pd.read_csv(filepath)
    sample_df = df.sample(n=sample_size, random_state=random_state)
    print(f"{len(sample_df)} queries from {filepath} have been loaded!!")
    return sample_df['query'].tolist()

def evaluate_model(model_name, queries, vector_search_engine, min_sim_threshold = 0.2, max_num_results = 5):
    """
    Evaluate a single model on the given list of queries and compute performance metrics.
    Metrics computed:
      1: Average top-1 similarity score (from the highest similarity result per query).
      2: Weighted average similarity score of the top 5 results (with re-normalized weights if fewer than 5 results are returned).
      3: Query coverage: Percentage of queries where the top result's similarity exceeds the threshold.
      4:: Average number of results returned.
      5: Average response time per query.
    """
    total_top_similarity = 0.0
    total_weighted_similarity = 0.0 
    total_response_time = 0.0
    coverage_count = 0
    total_results_count = 0
    
    num_queries = len(queries)
    detailed_results = []  # Storing per-query details

    # base weights for the top 5 results
    base_weights = [0.5, 0.2, 0.15, 0.1, 0.05]

    print(f"Starting evaluation for model: {model_name}")
    
    for idx, query in enumerate(queries):
        start_time = time.time()
        # Executing  search for the current query using the specified model.
        results = vector_search_engine.search(query, model_name, max_num_results=max_num_results, min_sim_threshold=min_sim_threshold)
        response_time = time.time() - start_time
        
        total_response_time += response_time

        # Calculating the top-1 similarity
        if not results.empty:
            top_similarity = results.iloc[0]['similarity_score']
            total_top_similarity += top_similarity
            
            # Considering and counting this query as covered if the top similarity meets the threshold: this is for the "query coverage" metric
            if top_similarity >= min_sim_threshold:
                coverage_count += 1

            total_results_count += len(results)
        else:
            top_similarity = 0
            total_results_count += 0
            total_top_similarity += 0

        # Calculating  the weighted average similarity for the top results.
        # Here, I am taking up to max_num_results (5); if fewer than 5, then re-normalize the weights (keeping sum = 1) for less than 5 results.
        if not results.empty:
            # similarity scores for the returned results (only taking top max_num_results)
            scores = results['similarity_score'].tolist()[:max_num_results]
            # Selecting the corresponding base weights
            selected_weights = base_weights[:len(scores)]
            # Re-normalize weights so they sum back to 1 for less than 5 results
            weight_sum = sum(selected_weights)
            normalized_weights = [w / weight_sum for w in selected_weights]
            # Calculating the weighted score
            weighted_score = sum(s * w for s, w in zip(scores, normalized_weights))
        else:
            weighted_score = 0

        total_weighted_similarity += weighted_score

        detailed_results.append({
            'query': query,
            'top_similarity': top_similarity,
            'weighted_top5_similarity': weighted_score,
            'num_results': len(results),
            'response_time': response_time
        })

        # just for debugging (if any issues occur later on) - debugging every 100 queries
        if (idx + 1) % 100 == 0 or (idx + 1) == num_queries:
            print(f"Model {model_name} - Processed {idx + 1}/{num_queries} queries. "
                  f"Current query top_similarity: {top_similarity:.4f}, "
                  f"Weighted Top-5: {weighted_score:.4f}, "
                  f"Results returned: {len(results)}, "
                  f"Response time: {response_time:.4f} sec.")

    avg_top_similarity = total_top_similarity / num_queries
    avg_weighted_similarity = total_weighted_similarity / num_queries
    avg_response_time = total_response_time / num_queries
    avg_num_results = total_results_count / num_queries
    coverage_percent = (coverage_count / num_queries) * 100

    metrics = {
        'model_name': model_name,
        'avg_top_similarity': avg_top_similarity,
        'avg_weighted_top5_similarity': avg_weighted_similarity,
        'coverage_percent': coverage_percent,
        'avg_num_results': avg_num_results,
        'avg_response_time': avg_response_time
    }
    
    print(f"Completed evaluation for model: {model_name}")
    print(f"Metrics: {metrics}\n")
    
    return metrics, detailed_results

def main():
    item_db_path = 'item_db.csv'
    item_searches_path = 'item_searches.csv'
    
    # Loading 1000 queries
    queries = load_queries(item_searches_path, sample_size=1000, random_state=42)
    
    # vector search engine initialization
    vector_search_engine = VectorSearchEngine(item_db_path)
    
    # taking out the available models in the registry
    registry = ModelRegistry()
    available_models = registry.available_models()  
    
    all_metrics = []  # for storing evaluation metrics for each model
    all_detailed_results = {}  #detailed results per model

    # Evaluating each model on the subset of 1000 queries 
    for model_name in available_models:
        print(f"\n--- Evaluating model: {model_name} ---")
        metrics, detailed_results = evaluate_model(
            model_name,
            queries,
            vector_search_engine,
            min_sim_threshold=0.2,
            max_num_results=5
        )
        all_metrics.append(metrics)
        all_detailed_results[model_name] = detailed_results
    
    # Creating a a CSV file of all metrics result for each model
    metrics_df = pd.DataFrame(all_metrics)
    metrics_df.to_csv('evaluation_metrics.csv', index=False)
    print("Aggregated metrics saved to 'evaluation_metrics.csv'.")
    
    # Also creating a txt file for the above.
    summary_lines = []
    summary_lines.append("------Model Evaluation Summary Report------\n------------\n")
    for metrics in all_metrics:
        summary_lines.append(f"Model: {metrics['model_name']}\n")
        summary_lines.append(f"  Average Top-1 Similarity: {metrics['avg_top_similarity']:.4f}\n")
        summary_lines.append(f"  Average Weighted Top-5 Similarity: {metrics['avg_weighted_top5_similarity']:.4f}\n")
        summary_lines.append(f"  Query Coverage (%): {metrics['coverage_percent']:.2f}\n")
        summary_lines.append(f"  Average Number of Results: {metrics['avg_num_results']:.2f}\n")
        summary_lines.append(f"  Average Response Time (s): {metrics['avg_response_time']:.4f}\n")
        summary_lines.append("------------\n")
    
    with open("evaluation_summary.txt", "w") as f:
        f.writelines(summary_lines)
    
    print("Summary report saved to 'evaluation_summary.txt'.")
    print("Evaluation complete.")

if __name__ == "__main__":
    main()