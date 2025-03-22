import pandas as pd
import time
from tqdm import tqdm
import numpy as np
from sentence_transformers import SentenceTransformer, util

class OpenSourceEmbeddingEvaluator:
    def __init__(self, model_name, item_db_path):
        self.model_name = model_name
        self.model = SentenceTransformer(model_name)
        self.df = pd.read_csv(item_db_path)
        self.name_embeddings = self.model.encode(self.df["item_name"].tolist(), normalize_embeddings=True)
        self.description_embeddings = self.model.encode(self.df["description"].tolist(), normalize_embeddings=True)

    def encode_query(self, query):
        return self.model.encode(query, normalize_embeddings=True)

    def search(self, query, max_num_results=5, min_sim_threshold=0.2):
        query_embedding = self.encode_query(query)
        name_sim = util.cos_sim(query_embedding, self.name_embeddings)[0].cpu().numpy()
        desc_sim = util.cos_sim(query_embedding, self.description_embeddings)[0].cpu().numpy()
        total_sim = 0.5 * name_sim + 0.5 * desc_sim

        self.df["similarity_score"] = total_sim
        self.df["name_similarity"] = name_sim
        self.df["description_similarity"] = desc_sim
        filtered_df = self.df[self.df["similarity_score"] >= min_sim_threshold]
        return filtered_df.sort_values(by="similarity_score", ascending=False).head(max_num_results)

    def evaluate(self, queries, max_num_results=5, min_sim_threshold=0.2):
        base_weights = [0.5, 0.2, 0.15, 0.1, 0.05]
        total_top1 = total_top5 = total_time = 0
        coverage_count = total_results = 0
        detailed_results = []

        for idx, query in enumerate(tqdm(queries, desc="Evaluating queries")):
            start = time.time()
            results = self.search(query, max_num_results, min_sim_threshold)
            elapsed = time.time() - start
            total_time += elapsed

            if not results.empty:
                top1 = results.iloc[0]["similarity_score"]
                total_top1 += top1
                if top1 >= min_sim_threshold:
                    coverage_count += 1
                total_results += len(results)
                scores = results["similarity_score"].tolist()
                weights = base_weights[:len(scores)]
                norm_weights = [w / sum(weights) for w in weights]
                weighted = sum(s * w for s, w in zip(scores, norm_weights))
            else:
                top1 = weighted = 0

            total_top5 += weighted

            detailed_results.append({
                'query': query,
                'top_similarity': top1,
                'weighted_top5_similarity': weighted,
                'num_results': len(results),
                'response_time': elapsed
            })

            if (idx + 1) % 100 == 0:
                print(f"Processed {idx + 1}/{len(queries)}")

        n = len(queries)
        return {
            'model_name': self.model_name,
            'avg_top_similarity': total_top1 / n,
            'avg_weighted_top5_similarity': total_top5 / n,
            'coverage_percent': (coverage_count / n) * 100,
            'avg_num_results': total_results / n,
            'avg_response_time': total_time / n
        }, detailed_results


# Run the evaluator
item_db_path = "item_db.csv"
query_file = "item_searches.csv"
sample_size = 1000
random_state = 42

df_queries = pd.read_csv(query_file).sample(n=sample_size, random_state=random_state)
queries = df_queries["query"].tolist()

evaluator = OpenSourceEmbeddingEvaluator("intfloat/e5-large-v2", item_db_path)
metrics, results = evaluator.evaluate(queries)

metrics_df = pd.DataFrame([metrics])
metrics_df.to_csv("evaluation_metrics_e5.csv", index=False)

summary_lines = [
    "------Model Evaluation Summary Report (E5-Large)------\n------------\n",
    f"Model: {metrics['model_name']}\n",
    f"  Average Top-1 Similarity: {metrics['avg_top_similarity']:.4f}\n",
    f"  Average Weighted Top-5 Similarity: {metrics['avg_weighted_top5_similarity']:.4f}\n",
    f"  Query Coverage (%): {metrics['coverage_percent']:.2f}\n",
    f"  Average Number of Results: {metrics['avg_num_results']:.2f}\n",
    f"  Average Response Time (s): {metrics['avg_response_time']:.4f}\n",
    "------------\n"
]

with open("evaluation_summary_e5.txt", "w") as f:
    f.writelines(summary_lines)

print("Evaluation complete. Metrics and summary saved.")