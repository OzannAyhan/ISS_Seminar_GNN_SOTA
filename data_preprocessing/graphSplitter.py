# Ali Jaabous
# Graph Splitter for Standard Split 80 - 10 - 10

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.transforms import RandomLinkSplit, ToUndirected

class GraphSplitter:
    def __init__(self):
        pass

    def split_data(self, data):
        torch.manual_seed(888)

        transform = RandomLinkSplit(
            num_val=0.1,
            num_test=0.1,
            key='edge_label',
            is_undirected=True,
            neg_sampling_ratio=0.0,
            add_negative_train_samples=False,
            edge_types=[('user', 'rating', 'movie')],
            rev_edge_types=[('movie', 'rev_rating', 'user')],
        )

        train_data, val_data, test_data = transform(data)

        ## Adjust data types
        train_data['user', 'rating', 'movie'].edge_label = train_data['user', 'rating', 'movie'].edge_label.double()
        val_data['user', 'rating', 'movie'].edge_label = val_data['user', 'rating', 'movie'].edge_label.double()

        train_data['user', 'rating', 'movie'].edge_attr = train_data['user', 'rating', 'movie'].edge_attr.double()
        val_data['user', 'rating', 'movie'].edge_attr = val_data['user', 'rating', 'movie'].edge_attr.double()

        return train_data, val_data, test_data

# this class isn't used in our main notebook but is attached to this repo to provide an
# alternative method to split the data (Standard Split)