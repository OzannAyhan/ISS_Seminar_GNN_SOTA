import pandas as pd
from node2vec import Node2Vec
import networkx as nx
#from movtit_nlp_v3 import MovieTitlesProcessor
from .movtit_nlp_v3 import MovieTitlesProcessor

class DataPreprocessor:
    def __init__(self, df, df_movie, cluster_amount=5, model_name="all-MiniLM-L6-v2", nlp_performed=False):
        self.df = df
        self.df_movie = df_movie
        self.cluster_amount = cluster_amount
        self.model_name = model_name
        self.nlp_performed = nlp_performed 

    def filter_data(self, min_userID=0, max_userID=100):
        self.df = self.df[(self.df['UserID'] >= min_userID) & (self.df['UserID'] <= max_userID)]
        self.ratings = self.df.copy()
        
    def create_rating_user_movie_info(self):
        ratings = self.df.copy()
        movies = self.df_movie.copy()
        userInfo = pd.DataFrame(self.df['UserID'].unique(), columns=['UserID'])
        movieInfo = pd.DataFrame(self.df['MovieID'].unique(), columns=['MovieID'])

        # Add movie_graph_id to ratings and movie_info
        movie_index_dict = {}
        for movie_id in movieInfo['MovieID']:
            if movie_id not in movie_index_dict.keys():
                movie_index_dict[movie_id] = len(movie_index_dict)
            else:
                print('Error - duplicate movie ids')

        ratings['movie_graph_id'] = ratings['MovieID'].apply(lambda x: movie_index_dict[x])
        movieInfo['movie_graph_id'] = movieInfo['MovieID'].apply(lambda x: movie_index_dict[x])

        movieInfo.sort_values(by=['movie_graph_id'], inplace=True)

        ratings['rating_count_per_user'] = ratings.groupby('UserID')['Rating'].transform('count')
        ratings['rating_count_per_movie'] = ratings.groupby('MovieID')['Rating'].transform('count')
        ratings['avg_rating_per_person'] = ratings.groupby('UserID')['Rating'].transform('mean').round(1)
        ratings['avg_rating_per_movie'] = ratings.groupby('MovieID')['Rating'].transform('mean').round(1)
        # Calculate ReleaseAge and merge into ratings
        movies = movies.dropna(subset=['Year'])
        latest_year = movies['Year'].max()
        movies['ReleaseAge'] = latest_year - movies['Year']
        ratings = ratings.merge(movies[['MovieID', 'ReleaseAge']], on='MovieID', how='left')
        
        # Merge ratings with additional info
        unique_columns_1 = ratings[['UserID', 'rating_count_per_user']].drop_duplicates().reset_index()
        unique_columns_2 = ratings[['UserID', 'avg_rating_per_person']].drop_duplicates().reset_index()
        unique_columns_3 = ratings[['MovieID', 'avg_rating_per_movie']].drop_duplicates().reset_index()
        unique_columns_4 = ratings[['MovieID', 'ReleaseAge']].drop_duplicates().reset_index()
        unique_columns_5 = ratings[['MovieID', 'rating_count_per_movie']].drop_duplicates().reset_index()


        userInfo = pd.concat([userInfo, unique_columns_1['rating_count_per_user'], unique_columns_2['avg_rating_per_person']], axis=1)
        movieInfo = pd.concat([movieInfo, unique_columns_3['avg_rating_per_movie'], unique_columns_4['ReleaseAge'],
                       unique_columns_5['rating_count_per_movie']], axis=1)


        self.userInfo = userInfo
        self.movieInfo = movieInfo
        self.ratings = ratings
        self.movies = movies

    def perform_nlp_tasks(self):
        movtit_nlp_instance = MovieTitlesProcessor(df=self.df_movie, cluster_amount=self.cluster_amount)
        self.clustered_movies = movtit_nlp_instance.process_movie_titles()
        self.nlp_performed = True
        self.movies = pd.merge(self.movies, self.clustered_movies[['MovieID', 'Cluster']], on='MovieID', how='left')
        # One-hot encoding for movie clusters
        one_hot_encodings = pd.get_dummies(self.movies['Cluster'], prefix='Cluster')
        self.movies = pd.concat([self.movies, one_hot_encodings], axis=1)
        self.ratings = pd.merge(self.ratings, self.movies[['MovieID', 'Cluster_0', 'Cluster_1', 'Cluster_2', 'Cluster_3', 'Cluster_4']], on='MovieID', how='left')
        # Merge ratings with clusters
        unique_columns_6 = self.ratings[['MovieID', 'Cluster_0']].drop_duplicates().reset_index()
        unique_columns_7 = self.ratings[['MovieID', 'Cluster_1']].drop_duplicates().reset_index()
        unique_columns_8 = self.ratings[['MovieID', 'Cluster_2']].drop_duplicates().reset_index()
        unique_columns_9 = self.ratings[['MovieID', 'Cluster_3']].drop_duplicates().reset_index()
        unique_columns_10 =self.ratings[['MovieID', 'Cluster_4']].drop_duplicates().reset_index()
        self.movieInfo = pd.concat([self.movieInfo, unique_columns_6['Cluster_0'], unique_columns_7['Cluster_1'],
                                    unique_columns_8['Cluster_2'], unique_columns_9['Cluster_3'], 
                                    unique_columns_10['Cluster_4']], axis=1)

    def get_node_embeddings_node2vec(self, dimensions=20, walk_length=16, num_walks=100, workers=4, window=10, min_count=1, batch_words=4):
        # Step 1: Prepare the Graph
        G = nx.Graph()
        for _, row in self.ratings.iterrows():
            G.add_node(row['UserID'], bipartite=0)  # Add user node
            G.add_node(row['MovieID'], bipartite=1)  # Add movie node
            G.add_edge(row['UserID'], row['MovieID'], weight=row['Rating'])

        # Step 2: Generate Node2Vec Embeddings
        node2vec = Node2Vec(G, dimensions=dimensions, walk_length=walk_length, num_walks=num_walks, workers=workers)
        model = node2vec.fit(window=window, min_count=min_count, batch_words=batch_words)

        # Extracting embeddings and converting node IDs back to integers
        user_embeddings = {int(node): model.wv[str(node)] for node in G.nodes() if G.nodes[node]['bipartite'] == 0 and str(node) in model.wv}
        movie_embeddings = {int(node): model.wv[str(node)] for node in G.nodes() if G.nodes[node]['bipartite'] == 1 and str(node) in model.wv}

        # Convert embeddings dictionaries to DataFrames
        user_embeddings_df = pd.DataFrame.from_dict(user_embeddings, orient='index').reset_index().rename(columns={'index': 'UserID'})
        movie_embeddings_df = pd.DataFrame.from_dict(movie_embeddings, orient='index').reset_index().rename(columns={'index': 'MovieID'})

        # Rename columns appropriately
        user_embeddings_df.columns = ['UserID'] + [f'user_embedding_{i}' for i in range(dimensions)]
        movie_embeddings_df.columns = ['MovieID'] + [f'movie_embedding_{i}' for i in range(dimensions)]

        # Ensure 'UserID' and 'MovieID' in 'ratings' are of type int
        self.ratings['UserID'] = self.ratings['UserID'].astype(int)
        self.ratings['MovieID'] = self.ratings['MovieID'].astype(int)

        # Step 3: Merge Embeddings with Ratings Data
        self.ratings = self.ratings.merge(user_embeddings_df, on='UserID', how='left').merge(movie_embeddings_df, on='MovieID', how='left')

        # Impute NaN values with 0
        self.ratings.fillna(0, inplace=True)
    
    
    
    def get_node_embeddings_deepwalk(self, dimensions=20, walk_length=16, num_walks=100, workers=4, window=10, min_count=1, batch_words=4):
        # Step 1: Prepare the Graph
        G = nx.Graph()

        # Adding nodes with the 'bipartite' attribute
        for _, row in self.ratings.iterrows():
            G.add_node(row['UserID'], bipartite=0)  # Add user node
            G.add_node(row['MovieID'], bipartite=1)  # Add movie node
            G.add_edge(row['UserID'], row['MovieID'], weight=row['Rating'])

        # Step 2: Generate DeepWalk Embeddings (using Node2Vec with p=1 and q=1)
        # Initialize Node2Vec model for DeepWalk with p=1 and q=1
        deepwalk = Node2Vec(G, dimensions=dimensions, walk_length=walk_length, num_walks=num_walks, workers=workers, p=1, q=1)

        # Train DeepWalk model
        model = deepwalk.fit(window=window, min_count=min_count, batch_words=batch_words)

        # Extracting embeddings and converting node IDs back to integers
        user_embeddings = {int(node): model.wv[str(node)] for node in G.nodes() if G.nodes[node]['bipartite'] == 0 and str(node) in model.wv}
        movie_embeddings = {int(node): model.wv[str(node)] for node in G.nodes() if G.nodes[node]['bipartite'] == 1 and str(node) in model.wv}

        # Convert embeddings dictionaries to DataFrames
        user_embeddings_df = pd.DataFrame.from_dict(user_embeddings, orient='index').reset_index()
        movie_embeddings_df = pd.DataFrame.from_dict(movie_embeddings, orient='index').reset_index()

        # Rename columns appropriately
        user_embeddings_df.columns = ['UserID'] + [f'user_embedding_{i}' for i in range(user_embeddings_df.shape[1] - 1)]
        movie_embeddings_df.columns = ['MovieID'] + [f'movie_embedding_{i}' for i in range(movie_embeddings_df.shape[1] - 1)]

        # Ensure 'UserID' and 'MovieID' in 'ratings' are of type int
        self.ratings['UserID'] = self.ratings['UserID'].astype(int)
        self.ratings['MovieID'] = self.ratings['MovieID'].astype(int)

        # Step 3: Merge Embeddings with Ratings Data
        ratings = self.ratings.merge(user_embeddings_df, on='UserID', how='left')
        self.ratings = ratings.merge(movie_embeddings_df, on='MovieID', how='left')

        # Impute NaN values with 0
        self.ratings.fillna(0, inplace=True)
    
 

# Usage example:
#from data_preprocess_v4 import DataPreprocessor
#df = pd.read_csv('/content/df_9500_17152.csv')
#df_movie = pd.read_excel('/content/movie_titles.xlsx')
#preprocessor = DataPreprocessor(df, df_movie)

#To filter our rating data
#preprocessor.filter_data(0,50)
#To see filtered data -->  preprocessor.ratings

#To get the new variables (avg, count, releaseage etc.) on ratings dataframe and to get movieInfo and userInfo dataframes
#preprocessor.create_rating_user_movie_info()
#To see new rating data --> preprocessor.ratings
#To see userInfo data --> preprocessor.userInfo
#To see movieInfo data --> preprocessor.movieInfo

#To apply NLP to movie_titles an get cluster dummies
#preprocessor.perform_nlp_tasks()
#To see new rating data --> preprocessor.ratings
#To see userInfo data --> preprocessor.userInfo
#To see movieInfo data --> preprocessor.movieInfo

#To apply node2vec in order to get node embeddings
#preprocessor.get_node_embeddings_node2vec()
#To see new rating data --> preprocessor.ratings

#To apply deepwalk in order to get node embeddings
#preprocessor.get_node_embeddings_deepwalk()
#To see new rating data --> preprocessor.ratings
