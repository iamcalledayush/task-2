"""
ItemDB Search Engine

This is a vector-based search engine designed to efficiently match user queries with items in a database.
The engine leverages sentence transformers for embedding generation providing fast and accurate results even on large datasets.

"""

import pandas as pd
import numpy as np
import gradio as gr
import time
from model_registry import ModelRegistry, ModelWrapper
# einops

class VectorSearchEngine:
    def __init__(self, csv_path):
        """
        Initialize the vector search engine

        :param csv_path: Path to the CSV file containing items and descriptions
        """
        # Load data
        self.df = pd.read_csv(csv_path)

        # Initialize embedding model registry
        self.model_registry = ModelRegistry()

        # Initialize default model
        self.device = "cpu"
        self.model_name = "all-MiniLM-L12-v2"
        self.model = self.model_registry.load(self.model_name, device=self.device)

        # Create vector database registry
        self.db_registry = self.create_vector_database_registry()

    def create_vector_database_registry(self):
        """
        Creates a vector database registry for all models in the model registry.

        For each model, it:
            1. Encodes item names separately.
            2. Encodes descriptions separately.
            3. Groups them in a dict.

        The resulting registry is stored as `self.db_registry`.

        Parameters:
            None

        Returns:
            dict: A dictionary containing vector embeddings for each model.
                Each entry in the dictionary has keys "name_embeddings" and "description_embeddings".
                These contain the encoded vectors for the item names and descriptions, respectively.
        """

        def create_vector_database(model: ModelWrapper):
            """
            Encode all items in the vector database and create a FAISS index
            """

            # Create embeddings of item names and descriptions
            name_embeddings = model.encode_data(self.df["item_name"].to_list(), normalize_embeddings=True)
            description_embeddings = model.encode_data(self.df["description"].to_list(), normalize_embeddings=True)

            return {"name_embeddings": name_embeddings, "description_embeddings": description_embeddings}

        # Create vector databases for each model
        db_registry = {}
        for model_name in self.model_registry.available_models():
            print(f"Processing the DB with model: {model_name}...")
            model = self.model_registry.load(model_name=model_name, device=self.device)
            db_registry[model_name] = create_vector_database(model)
        return db_registry

    def search(
            self,
            query: str,
            model_name: str,
            max_num_results: int,
            min_sim_threshold: float,
    ) -> pd.DataFrame:
        """
        Search the vector database for items most similar to the provided query.

        Parameters:
            query (str): A string representing the search query.
            model_name (str): The name of the model used for encoding and searching.
            max_num_results (int, optional): The number of top results to return.
            min_sim_threshold (float, optional): The minimum similarity score required
                                                for a result to be included.

        Returns:
            pd.DataFrame: A DataFrame containing the top similar items and their similarity scores.

        Notes:
            - The similarity threshold filters out results that do not meet the specified minimum similarity.
            - Ensure the query_embedding is correctly normalized before calling this method.
        """

        # Configure model
        if self.model_name != model_name:
            print(f"Model changed. Loading model {model_name}... (it might take some time)")
            self.model_name = model_name
            self.model: ModelWrapper = self.model_registry.load(model_name=model_name, device=self.device)

        # Encode the query and ensure float32
        query_embedding = self.model.encode_query([query], normalize_embeddings=True)

        # Compute similarity for names and descriptions
        # TODO abstract into separate class to support configured granular computation
        name_similarities = np.dot(query_embedding, self.db_registry[model_name]["name_embeddings"].T).flatten()
        description_similarities = np.dot(query_embedding,
                                          self.db_registry[model_name]["description_embeddings"].T).flatten()

        # Combine similarities and attach to text df
        similar_items = self.df.copy()
        similar_items["similarity_score"] = 0.5 * name_similarities + 0.5 * description_similarities
        similar_items["name_similarity"] = name_similarities
        similar_items["description_similarity"] = description_similarities

        # Retrieve the top results
        similar_items = similar_items.sort_values(by="similarity_score", ascending=False).head(max_num_results)

        # Filter based on a similarity threshold
        similar_items = similar_items[similar_items["similarity_score"] >= min_sim_threshold]

        return similar_items

    def create_gradio_interface(self):
        """
        Create a Gradio interface for the search engine.

        This function sets up a user-friendly interface that allows users to input their query,
        specify the number of results they want to see, and set a minimum similarity threshold.
        The interface then calls the `search` method of the search engine with these inputs
        and displays the results along with any relevant messages or warnings.

        Parameters:
            None (the function is called as part of a class method)

        Returns:
            A Gradio Interface object that can be displayed to users for interacting with the search engine.
        """

        def search_wrapper(query: str, model_name: str, max_num_results: int, min_sim_threshold: float):
            """
            Wrapper function for handling the search request and formatting the results.

            Parameters:
                query (str): The user's query describing the item they want to donate.
                model_name (str): The name of the desired model to be used for similarity computation.
                max_num_results (int): The number of top-k results to return.
                min_sim_threshold (float): The minimum similarity score required.

            Returns:
                tuple: A tuple containing the formatted results text and a warning message.
            """

            try:
                # Record the start time for performance measurement
                ts = time.perf_counter()

                # Perform the search using the provided query, number of results, and similarity threshold
                results = self.search(
                    query=query,
                    model_name=model_name,
                    max_num_results=max_num_results,
                    min_sim_threshold=min_sim_threshold
                )

                # Check if the number of results is less than 3 or more than 10 to provide feedback
                if len(results) < 3:
                    message = "Your query might be too specific, try a more generic description."
                elif len(results) > 10:
                    message = "Your query might be too generic, try a more specific description."
                else:
                    message = "Everything looks fine. Here are your results:"

                # Format the results for display
                result_text = ""
                for _, row in results.iterrows():
                    result_text += f"Item: {row['item_name']}   |   Score: {row['similarity_score']:.2f}    "
                    result_text += f"({row['name_similarity']:.2f}, {row['description_similarity']:.2f})\n"
                    result_text += f"Description: {row['description']}\n\n"

                # Calculate and print the elapsed time
                elapsed_time = time.perf_counter() - ts
                print(f"Search took {elapsed_time:.4f} seconds")

                return (message, result_text)

            except Exception as e:
                # Handle any exceptions that occur during the search process
                return (f"An error occurred: {str(e)}", "")

        # Create Gradio interface with defined input and output components

        iface = gr.Interface(
            fn=search_wrapper,
            inputs=[
                gr.Textbox(
                    label="Query",
                    info="Describe briefly the item you would like to donate."
                ),
                gr.Dropdown(
                    self.model_registry.available_models(),
                    label="Model", value="all-MiniLM-L12-v2"
                ),
                gr.Number(
                    label="Maximum Number of Results",
                    value=5,
                    precision=0,
                    minimum=1,
                    maximum=100,
                    step=1,
                    info="Enter number of items to display (1-100)."
                ),
                gr.Number(
                    label="Minimum Similarity Score",
                    value=0.2,
                    minimum=0.0,
                    maximum=1.0,
                    step=0.01,
                    info="Enter minimum similarity threshold (0-1)."
                )
            ],
            outputs=[
                gr.Textbox(label="Message for the User"),
                gr.Textbox(label="Search Results", info="Here are some item categories we suggest you choose:"),
            ],
            title="Item Search Engine",
            theme=gr.themes.Default(primary_hue="red", secondary_hue="red", neutral_hue="blue"),
            allow_flagging="never"
        )

        return iface


def main():
    """
    This function runs the gradio server for the ItemDB Search Engine

    You can access it in your browser at: http://localhost:7860/

    Example item_db.csv:
        item_name,description
        Sectional Sofa,Large modular seating arrangement with multiple connected pieces for spacious living rooms
        Shoe Cabinet,Slim storage furniture with multiple compartments for organized shoe collection
        Modern Dining Table,Minimalist rectangular table with sleek metal legs for contemporary dining spaces

    For a db of ~100 items the code runs in about 10 seconds on a higher end consumer GPU or CPU.

    Example query: I have an old shoe shelf in my basement
        - should return Shoe Cabinet as the top item
        - runs in under 20 ms even on a CPU
    """
    # Create the search engine from the ItemDB csv
    # search_engine = VectorSearchEngine(r"path\to\item_db.csv")
    search_engine = VectorSearchEngine(
        r"item_db.csv")

    # Create and launch the UI
    interface = search_engine.create_gradio_interface()
    interface.launch()


if __name__ == "__main__":
    main()
