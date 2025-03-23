# import pandas as pd
# import time
# import os
# import openai
# import json
# from dotenv import load_dotenv
# from tqdm import tqdm
# from sentence_transformers import SentenceTransformer, util



#--------Following commented code is to run the GPT4o API to generate enhanced detailed queries using STEPBACK PROMPTING method--------

# # Load environment variables
# load_dotenv()
# client = openai.OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))

# # ---------- CONFIG ----------
# ITEM_DB_PATH = "item_db.csv"
# QUERY_FILE = "item_searches.csv"
# MODEL_NAME = "intfloat/e5-large-v2"
# SAMPLE_SIZE = 1000
# RANDOM_STATE = 42

# # ---------- STEP 1: LOAD AND SAMPLE QUERIES ----------
# df_queries = pd.read_csv(QUERY_FILE).sample(n=SAMPLE_SIZE, random_state=RANDOM_STATE)
# original_queries = df_queries["query"].tolist()

# # ---------- STEP 2: STEP-BACK QUERY GENERATION ----------
# def generate_step_back_query(original_query):
#     prompt = f"""
#     You are an expert in reformulating user queries into more general questions that better capture the user's intent. 

#     Your task is to generate a *step-back query*, which is a more abstract and generalized version of the original query. The step-back query should:
#     •⁠  ⁠*Preserve the core meaning* of the original query.
#     •⁠  ⁠*Uncover implicit sub-questions* that the user might be asking.
#     •⁠  ⁠*Rephrase specific details into broader concepts* where appropriate.
#     •⁠  ⁠*Make retrieval easier by bridging the gap between the user's query and the information they seek.*
#     •⁠  ⁠*Support multi-sentence reformulations* if necessary.

#     Now, generate a step-back query for the following user input:
#     Original Query: "{original_query}"

#     Return only the result in the following JSON format:

#     {{ "step_back_query": "<your step-back reformulated query here>" }}
#     """
#     try:
#         response = client.chat.completions.create(
#             model="gpt-4o",
#             messages=[{"role": "user", "content": prompt}],
#             temperature=0.3,
#             max_tokens=200
#         )
#         content = response.choices[0].message.content.strip()

#         if content.startswith("```"):
#             content = content.strip("`").strip()
#             if content.lower().startswith("json"):
#                 content = content[4:].strip()

#         parsed = json.loads(content)
#         return parsed.get("step_back_query", "")
#     except Exception as e:
#         print(f"Error processing query: {original_query}\nException: {e}")
#         return ""

# # Generate enhanced queries
# enhanced_queries = []
# for query in tqdm(original_queries, desc="Enhancing queries with GPT-4o"):
#     enhanced = generate_step_back_query(query)
#     enhanced_queries.append(enhanced)
#     time.sleep(0.5)

# # Save enhanced queries
# df_enhanced = pd.DataFrame({
#     "original_query": original_queries,
#     "enhanced_query": enhanced_queries
# })
# df_enhanced.to_csv("stepback_enhanced_queries_sampled.csv", index=False)
# print("Saved step-back enhanced queries to stepback_enhanced_queries_sampled.csv")





# Following code is to evaluate the e5-large-v2 model on STEPBACK queries averaged with original queries

import pandas as pd
import time
from tqdm import tqdm
import numpy as np
from sentence_transformers import SentenceTransformer, util

class E5AveragedEmbeddingEvaluator:
    def __init__(self, model_name, item_db_path):
        self.model_name = model_name
        self.model = SentenceTransformer(model_name)
        self.df = pd.read_csv(item_db_path)
        self.name_embeddings = self.model.encode(self.df["item_name"].tolist(), normalize_embeddings=True)
        self.description_embeddings = self.model.encode(self.df["description"].tolist(), normalize_embeddings=True)

    def encode_average_query(self, query1, query2):
        emb1 = self.model.encode(query1, normalize_embeddings=True)
        emb2 = self.model.encode(query2, normalize_embeddings=True)
        return (emb1 + emb2) / 2

    def search(self, query_embedding, max_num_results=5, min_sim_threshold=0.2):
        name_sim = util.cos_sim(query_embedding, self.name_embeddings)[0].cpu().numpy()
        desc_sim = util.cos_sim(query_embedding, self.description_embeddings)[0].cpu().numpy()
        total_sim = 0.5 * name_sim + 0.5 * desc_sim

        self.df["similarity_score"] = total_sim
        self.df["name_similarity"] = name_sim
        self.df["description_similarity"] = desc_sim
        filtered_df = self.df[self.df["similarity_score"] >= min_sim_threshold]
        return filtered_df.sort_values(by="similarity_score", ascending=False).head(max_num_results)

    def evaluate(self, original_queries, enhanced_queries, max_num_results=5, min_sim_threshold=0.2):
        base_weights = [0.5, 0.2, 0.15, 0.1, 0.05]
        total_top1 = total_top5 = total_time = 0
        coverage_count = total_results = 0
        detailed_results = []

        for idx, (orig, enhanced) in enumerate(tqdm(zip(original_queries, enhanced_queries), total=len(original_queries))):
            start = time.time()
            avg_embedding = self.encode_average_query(orig, enhanced)
            results = self.search(avg_embedding, max_num_results, min_sim_threshold)
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
                'original_query': orig,
                'enhanced_query': enhanced,
                'top_similarity': top1,
                'weighted_top5_similarity': weighted,
                'num_results': len(results),
                'response_time': elapsed
            })

            if (idx + 1) % 100 == 0:
                print(f"Processed {idx + 1}/{len(original_queries)}")

        n = len(original_queries)
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
query_csv = "stepback_enhanced_queries_sampled.csv" 

df_queries = pd.read_csv(query_csv)
original_queries = df_queries["original_query"].tolist()
enhanced_queries = df_queries["enhanced_query"].tolist()

evaluator = E5AveragedEmbeddingEvaluator("intfloat/e5-large-v2", item_db_path)
metrics, results = evaluator.evaluate(original_queries, enhanced_queries)

# Save outputs
pd.DataFrame([metrics]).to_csv("evaluation_metrics_e5_avg_embeddings.csv", index=False)
pd.DataFrame(results).to_csv("evaluation_details_e5_avg_embeddings.csv", index=False)

# Print summary
print("\n------ Evaluation Summary (E5 with Averaged Embeddings) ------")
for k, v in metrics.items():
    print(f"{k.replace('_', ' ').title()}: {v:.4f}" if isinstance(v, float) else f"{k}: {v}")