# Ali Jaabous
# DataToGraph

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.transforms import RandomLinkSplit, ToUndirected

class DataToGraph:
    def __init__(self, ratings, userInfo, movieInfo):
        self.ratings = ratings
        self.userInfo = userInfo
        self.movieInfo = movieInfo

    def convert_to_hetero_data(self):
        user_ids = np.array(self.ratings['UserID'])
        movie_ids = np.array(self.ratings['movie_graph_id'])
        rating_ids = np.array(self.ratings['Rating'])

        data = HeteroData()

        # Node features
        data['user'].x = torch.tensor(np.array(self.userInfo.drop(columns=['UserID'])))
        data['movie'].x = torch.tensor(np.array(self.movieInfo.drop(columns=['MovieID', 'movie_graph_id'])))

        # Edges (simultaneously user and movie nodes, already with their types)
        data['user', 'rating', 'movie'].edge_index = torch.stack([torch.tensor(user_ids, dtype=torch.long),
                                                                  torch.tensor(movie_ids, dtype=torch.long)])

        # Edge weights
        data['user', 'rating', 'movie'].edge_label = torch.tensor(rating_ids, dtype=torch.double)
        data['user', 'rating', 'movie'].edge_attr = torch.tensor(rating_ids, dtype=torch.double)

        data = ToUndirected()(data)

        return data

# this class isn't used in our main notebook but is attached to this repo to provide an
# alternative method to create graph-structured data
