#Ali Jaabous
#GraphDataBuilder2Fold

import numpy as np
import torch
from torch_geometric.data import HeteroData
from torch_geometric.transforms import ToUndirected

class GraphDataBuilder2Fold:

    def __init__(self, userInfo, movieInfo, ratings):
        self.userInfo = userInfo
        self.movieInfo = movieInfo
        self.ratings = ratings
        self.data = HeteroData()
        self.train_data_1 = HeteroData()
        self.train_data_2 = HeteroData()
        self.val_data_1 = HeteroData()
        self.val_data_2 = HeteroData()
        self.train_data_final = HeteroData()
        self.test_data = HeteroData()

    def build_data(self):
        # Extracting user, movie, and rating IDs
        user_ids = np.array(self.ratings['UserID'])
        movie_ids = np.array(self.ratings['movie_graph_id'])
        rating_ids = np.array(self.ratings['Rating'])

        # Node features
        self.data['user'].x = torch.tensor(np.array(self.userInfo.drop(columns=['UserID'])))
        self.data['movie'].x = torch.tensor(np.array(self.movieInfo.drop(columns=['MovieID', 'movie_graph_id'])))

        # Edges
        edge_index = torch.stack([torch.tensor(user_ids, dtype=torch.long),
                                  torch.tensor(movie_ids, dtype=torch.long)])

        # Edge weights
        edge_weights = torch.tensor(rating_ids, dtype=torch.double)

        self.data['user', 'rating', 'movie'].edge_index = edge_index
        self.data['user', 'rating', 'movie'].edge_label = edge_weights
        self.data['user', 'rating', 'movie'].edge_attr = edge_weights

        # To undirected
        self.data = ToUndirected()(self.data)

        # This is my train_data Fold 1
        # Node features
        self.train_data_1['user'].x = torch.tensor(np.array(self.userInfo.drop(columns=['UserID'])))
        self.train_data_1['movie'].x = torch.tensor(np.array(self.movieInfo.drop(columns=['MovieID', 'movie_graph_id'])))

        # Edges (simultaneously user and movie nodes, already with their types)
        train_edge_object_1 = torch.stack([torch.tensor(user_ids, dtype=torch.long), torch.tensor(movie_ids, dtype=torch.long)])
        self.train_data_1['user', 'rating', 'movie'].edge_index = train_edge_object_1[:, :(int(train_edge_object_1.shape[1] * (1-0.45-0.1)))]

        # Edge weights
        self.train_data_1['user', 'rating', 'movie'].edge_label = torch.tensor(rating_ids[:int((len(rating_ids)) * (1-0.45-0.1))], dtype=torch.double)
        self.train_data_1['user', 'rating', 'movie'].edge_attr = torch.tensor(rating_ids[:int((len(rating_ids)) * (1-0.45-0.1))], dtype=torch.double)

        train_edge_label_object_1 = torch.stack([torch.tensor(user_ids, dtype=torch.long), torch.tensor(movie_ids, dtype=torch.long)])
        self.train_data_1['user', 'rating', 'movie'].edge_label_index = train_edge_label_object_1[:,:(int(train_edge_label_object_1.shape[1] * (1-0.45-0.1)))]

        # Edges (simultaneously user and movie nodes, already with their types)
        train_rev_edge_object_1 = torch.stack([torch.tensor(movie_ids, dtype=torch.long), torch.tensor(user_ids, dtype=torch.long)])
        self.train_data_1['movie', 'rev_rating', 'user'].edge_index = train_rev_edge_object_1[:, :(int(train_rev_edge_object_1.shape[1] * (1-0.45-0.1)))]

        # Edge weights
        self.train_data_1['movie', 'rev_rating', 'user'].edge_label = torch.tensor(rating_ids[:int((len(rating_ids)) * (1-0.45-0.1))], dtype=torch.double)
        self.train_data_1['movie', 'rev_rating', 'user'].edge_attr = torch.tensor(rating_ids[:int((len(rating_ids)) * (1-0.45-0.1))], dtype=torch.double)

        # This is my train_data Fold 2
        # Node features
        self.train_data_2['user'].x = torch.tensor(np.array(self.userInfo.drop(columns=['UserID'])))
        self.train_data_2['movie'].x = torch.tensor(np.array(self.movieInfo.drop(columns=['MovieID', 'movie_graph_id'])))

        # Edges (simultaneously user and movie nodes, already with their types)
        train_edge_object_2 = torch.stack([torch.tensor(user_ids, dtype=torch.long), torch.tensor(movie_ids, dtype=torch.long)])
        self.train_data_2['user', 'rating', 'movie'].edge_index = train_edge_object_2[:, (int(train_edge_object_2.shape[1] * (0.45))):(int(train_edge_object_2.shape[1] * (1-0.1)))]

        # Edge weights
        self.train_data_2['user', 'rating', 'movie'].edge_label = torch.tensor(rating_ids[int((len(rating_ids)) * 0.45):int((len(rating_ids)) * (1-0.1))], dtype=torch.double)
        self.train_data_2['user', 'rating', 'movie'].edge_attr = torch.tensor(rating_ids[int((len(rating_ids)) * 0.45):int((len(rating_ids)) * (1-0.1))], dtype=torch.double)

        train_edge_label_object_2 = torch.stack([torch.tensor(user_ids, dtype=torch.long), torch.tensor(movie_ids, dtype=torch.long)])
        self.train_data_2['user', 'rating', 'movie'].edge_label_index = train_edge_label_object_2[:, (int(train_edge_label_object_2.shape[1] * 0.45)):(int(train_edge_label_object_2.shape[1] * (1-0.1)))]

        # Edges (simultaneously user and movie nodes, already with their types)
        train_rev_edge_object_2 = torch.stack([torch.tensor(movie_ids, dtype=torch.long), torch.tensor(user_ids, dtype=torch.long)])
        self.train_data_2['movie', 'rev_rating', 'user'].edge_index = train_rev_edge_object_2[:, (int(train_rev_edge_object_2.shape[1] * 0.45)):(int(train_rev_edge_object_2.shape[1] * (1-0.1)))]

        # Edge weights
        self.train_data_2['movie', 'rev_rating', 'user'].edge_label = torch.tensor(rating_ids[int((len(rating_ids)) * 0.45):int((len(rating_ids)) * (1-0.1))], dtype=torch.double)
        self.train_data_2['movie', 'rev_rating', 'user'].edge_attr = torch.tensor(rating_ids[int((len(rating_ids)) * 0.45):int((len(rating_ids)) * (1-0.1))], dtype=torch.double)

        # This is my final train_data
        # Node features
        self.train_data_final['user'].x = torch.tensor(np.array(self.userInfo.drop(columns=['UserID'])))
        self.train_data_final['movie'].x = torch.tensor(np.array(self.movieInfo.drop(columns=['MovieID', 'movie_graph_id'])))

        # Edges (simultaneously user and movie nodes, already with their types)
        train_edge_object_final = torch.stack([torch.tensor(user_ids, dtype=torch.long), torch.tensor(movie_ids, dtype=torch.long)])
        self.train_data_final['user', 'rating', 'movie'].edge_index = train_edge_object_final[:, :(int(train_edge_object_final.shape[1] * (1-0.1)))]

        # Edge weights
        self.train_data_final['user', 'rating', 'movie'].edge_label = torch.tensor(rating_ids[:int((len(rating_ids)) * (1-0.1))], dtype=torch.double)
        self.train_data_final['user', 'rating', 'movie'].edge_attr = torch.tensor(rating_ids[:int((len(rating_ids)) * (1-0.1))], dtype=torch.double)

        train_edge_label_object_final = torch.stack([torch.tensor(user_ids, dtype=torch.long), torch.tensor(movie_ids, dtype=torch.long)])
        self.train_data_final['user', 'rating', 'movie'].edge_label_index = train_edge_label_object_final[:, :(int(train_edge_label_object_final.shape[1] * (1-0.1)))]

        # Edges (simultaneously user and movie nodes, already with their types)
        train_rev_edge_object_final = torch.stack([torch.tensor(movie_ids, dtype=torch.long), torch.tensor(user_ids, dtype=torch.long)])
        self.train_data_final['movie', 'rev_rating', 'user'].edge_index = train_rev_edge_object_final[:, :(int(train_rev_edge_object_final.shape[1] * (1-0.1)))]

        # Edge weights
        self.train_data_final['movie', 'rev_rating', 'user'].edge_label = torch.tensor(rating_ids[:int((len(rating_ids)) * (1-0.1))], dtype=torch.double)
        self.train_data_final['movie', 'rev_rating', 'user'].edge_attr = torch.tensor(rating_ids[:int((len(rating_ids)) * (1-0.1))], dtype=torch.double)

        # This is my test_data
        # Node features
        self.test_data['user'].x = torch.tensor(np.array(self.userInfo.drop(columns=['UserID'])))
        self.test_data['movie'].x = torch.tensor(np.array(self.movieInfo.drop(columns=['MovieID', 'movie_graph_id'])))

        # Edges (simultaneously user and movie nodes, already with their types)
        test_edge_object = torch.stack([torch.tensor(user_ids, dtype=torch.long), torch.tensor(movie_ids, dtype=torch.long)])
        self.test_data['user', 'rating', 'movie'].edge_index = test_edge_object[:, :(int(test_edge_object.shape[1] * (1-0.1)))]

        # Edge weights
        self.test_data['user', 'rating', 'movie'].edge_label = torch.tensor(rating_ids[int((len(rating_ids)) * (1-0.1)):], dtype=torch.double)
        self.test_data['user', 'rating', 'movie'].edge_attr = torch.tensor(rating_ids[:int((len(rating_ids)) * (1-0.1))], dtype=torch.double)

        test_edge_label_object = torch.stack([torch.tensor(user_ids, dtype=torch.long), torch.tensor(movie_ids, dtype=torch.long)])
        self.test_data['user', 'rating', 'movie'].edge_label_index = test_edge_label_object[:, (int(test_edge_label_object.shape[1] * (1-0.1))):]

        # Edges (simultaneously user and movie nodes, already with their types)
        test_rev_edge_object = torch.stack([torch.tensor(movie_ids, dtype=torch.long), torch.tensor(user_ids, dtype=torch.long)])
        self.test_data['movie', 'rev_rating', 'user'].edge_index = test_rev_edge_object[:, :(int(test_rev_edge_object.shape[1] * (1-0.1)))]

        # Edge weights
        self.test_data['movie', 'rev_rating', 'user'].edge_label = torch.tensor(rating_ids[:int((len(rating_ids)) * (1-0.1))], dtype=torch.double)
        self.test_data['movie', 'rev_rating', 'user'].edge_attr = torch.tensor(rating_ids[:int((len(rating_ids)) * (1-0.1))], dtype=torch.double)


        return self.data, self.train_data_1, self.train_data_2, self.train_data_2, self.train_data_1, self.train_data_final, self.test_data

# The function of this class is to transform the data provided by the data_precessor into 
# graph structured data and split it by means of 2-Fold Further the final training set for 
# the evaluation is provided
