import pandas as pd
import numpy as np
import gradio as gr
import time
from model_registry import ModelRegistry, ModelWrapper


class VectorSearchEngine:
    def __init__(self, csv_path):
        self.df = pd.read_csv(csv_path)
        self.model_registry = ModelRegistry()
        self.device = "cpu"
        self.model_name = "all-MiniLM-L12-v2"
        self.model = self.model_registry.load(self.model_name, device=self.device)
        self.db_registry = self.create_vector_database_registry()

    def create_vector_database_registry(self):
        def create_vector_database(model: ModelWrapper):
            # Combine fields into a single rich string per item
            full_texts = (
                self.df['item_name'].fillna('') + '. ' +
                self.df['description'].fillna('') + '. ' +
                self.df['category'].fillna('') + '. ' +
                self.df['intended_use'].fillna('') + '. ' +
                self.df['indoor_or_outdoor'].fillna('') + '. ' +
                self.df['portable'].fillna('') + '. ' +
                self.df['size'].fillna('')
            )
            combined_embeddings = model.encode_data(full_texts.tolist(), normalize_embeddings=True)
            return {"combined_embeddings": combined_embeddings}

        db_registry = {}
        for model_name in self.model_registry.available_models():
            print(f"Processing the DB with model: {model_name}...")
            model = self.model_registry.load(model_name=model_name, device=self.device)
            db_registry[model_name] = create_vector_database(model)
        return db_registry

    def search(self, query: str, model_name: str, max_num_results: int, min_sim_threshold: float) -> pd.DataFrame:
        if self.model_name != model_name:
            print(f"Model changed. Loading model {model_name}...")
            self.model_name = model_name
            self.model: ModelWrapper = self.model_registry.load(model_name=model_name, device=self.device)

        query_embedding = self.model.encode_query([query], normalize_embeddings=True)
        combined_similarities = np.dot(query_embedding, self.db_registry[model_name]["combined_embeddings"].T).flatten()

        results = self.df.copy()
        results["similarity_score"] = combined_similarities
        results = results.sort_values(by="similarity_score", ascending=False).head(max_num_results)
        results = results[results["similarity_score"] >= min_sim_threshold]

        return results

    def create_gradio_interface(self):
        def search_wrapper(query: str, model_name: str, max_num_results: int, min_sim_threshold: float):
            try:
                ts = time.perf_counter()
                results = self.search(query, model_name, max_num_results, min_sim_threshold)
                message = "Here are your results:"
                if len(results) < 3:
                    message = "Your query might be too specific, try a more generic description."
                elif len(results) > 10:
                    message = "Your query might be too generic, try a more specific description."

                result_text = ""
                for _, row in results.iterrows():
                    result_text += f"Item: {row['item_name']}   |   Score: {row['similarity_score']:.2f}\n"
                    result_text += f"Description: {row['description']}\n\n"

                elapsed_time = time.perf_counter() - ts
                print(f"Search took {elapsed_time:.4f} seconds")
                return (message, result_text)

            except Exception as e:
                return (f"An error occurred: {str(e)}", "")

        iface = gr.Interface(
            fn=search_wrapper,
            inputs=[
                gr.Textbox(label="Query", info="Describe the item you want to donate."),
                gr.Dropdown(self.model_registry.available_models(), label="Model", value="all-MiniLM-L12-v2"),
                gr.Number(label="Max Results", value=5, minimum=1, maximum=100, step=1),
                gr.Number(label="Min Similarity", value=0.2, minimum=0.0, maximum=1.0, step=0.01)
            ],
            outputs=[
                gr.Textbox(label="Message"),
                gr.Textbox(label="Search Results")
            ],
            title="Enhanced Item Search Engine",
            theme=gr.themes.Default(primary_hue="red", secondary_hue="red", neutral_hue="blue"),
            allow_flagging="never"
        )
        return iface

def main():
    search_engine = VectorSearchEngine("item_db_enriched.csv")
    interface = search_engine.create_gradio_interface()
    interface.launch()

if __name__ == "__main__":
    main()
