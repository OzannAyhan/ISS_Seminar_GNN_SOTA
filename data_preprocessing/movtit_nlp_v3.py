from sentence_transformers import SentenceTransformer
from sklearn.cluster import KMeans
import pandas as pd
import matplotlib.pyplot as plt

class MovieTitlesProcessor:
    def __init__(self, df, cluster_amount, model_name="all-MiniLM-L6-v2"):
        super().__init__()
        self.df = df
        self.cluster_amount = cluster_amount
        self.model_name = model_name
        self.clustered_movies = None
        self.kmeans = None          
        self.movie_embeddings_cpu = None

    def process_movie_titles(self):
        # Load a pre-trained SentenceBERT model
        model = SentenceTransformer(self.model_name)

        # Step 1: Encode movie titles using SentenceBERT
        movie_embeddings = model.encode(self.df['Title'].astype(str).tolist(), convert_to_tensor=True)
        movie_embeddings_cpu = movie_embeddings.cpu().numpy()  

        # Step 2: Clustering (K-Means)
        kmeans = KMeans(n_clusters=self.cluster_amount, random_state=42)
        clusters = kmeans.fit_predict(movie_embeddings_cpu)
        
        # Create a DataFrame with clustered movies
        self.clustered_movies = pd.DataFrame({"MovieID": self.df['MovieID'], "Title": self.df['Title'], "Cluster": clusters, "Distance": kmeans.transform(movie_embeddings_cpu).min(axis=1)})
        self.kmeans = kmeans
        self.movie_embeddings_cpu = movie_embeddings_cpu

        return self.clustered_movies

# Example usage:
# Assuming you have a DataFrame named 'movie_titles'
# First run this --> movtit_nlp_instance = MovieTitlesProcessor(df=movie_titles, cluster_amount=5)
# To store the clustered movie results, run this --> clustered_movies_result = movtit_nlp_instance.process_movie_titles()
# print(clustered_movies_result)
