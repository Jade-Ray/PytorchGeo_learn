# %% [markdown]
# # A node classification task in `Zachary’s Karate Club` Dataset
#
# *reffered by this [blog](https://towardsdatascience.com/a-beginners-guide-to-graph-neural-networks-using-pytorch-geometric-part-1-d98dc93e7742)*
#
# ---
#
# In order to formulate the problem, we need to:
# 1. The graph itself and the labels for each node
# 2. The edge data in the Coordinate Format(COO)
# 3. Embeddings or numerical representations for the nodes, *used by node degree*

# %%
import networkx as nx
import numpy as np
import torch
from sklearn.preprocessing import StandardScaler

# load graph from networkx library
G = nx.karate_club_graph()

# retrieve the labels for each node
labels = np.asarray([G.nodes[i]['club'] != 'Mr. Hi' for i in G.nodes]).astype(np.int64)

# create edge index from
adj = nx.to_scipy_sparse_matrix(G, format='coo')
row = torch.from_numpy(adj.row.astype(np.int64)).to(torch.long)
col = torch.from_numpy(adj.col.astype(np.int64)).to(torch.long)
edge_index = torch.stack([row, col], dim=0)

# using degree as embedding
embeddings = np.array(list(dict(G.degree()).values()))

# normalizing degree values
scale = StandardScaler()
embeddings = scale.fit_transform(embeddings.reshape(-1, 1))

# %% [markdown]
# #### The Custom dataset

# %%
import torch
import pandas as pd
from torch_geometric.data import InMemoryDataset, Data
from sklearn.model_selection import train_test_split
import torch_geometric.transforms as T

# custom dataset
class KarateDataset(InMemoryDataset):
    def __init__(self, transform=None):
        super(KarateDataset, self).__init__('.', transform, None, None)
        
        data = Data(edge_index=edge_index)
        
        data.num_nodes = G.number_of_nodes()
        
        # embedding
        data.x = torch.from_numpy(embeddings).type(torch.float32)
        
        # labels
        y = torch.from_numpy(labels).type(torch.long)
        data.y = y.clone().detach()
        
        data.num_classes = 2
        
        # splitting the data into train, validation and test
        X_train, X_test, y_train, y_test = train_test_split(pd.Series(G.nodes()), 
                                                            pd.Series(labels),
                                                            test_size=0.30,
                                                            random_state=42)
        
        n_nodes = G.number_of_nodes()
        
        # create train and test masks for data
        train_mask = torch.zeros(n_nodes, dtype=torch.bool)
        test_mask = torch.zeros(n_nodes, dtype=torch.bool)
        train_mask[X_train.index] = True
        test_mask[X_test.index] = True
        data['train_mask'] = train_mask
        data['test_mask'] = test_mask 
        
        self.data, self.slices = self.collate([data])
        
    def _download(self):
        return

    def _process(self):
        return

    def __repr__(self):
        return '{}()'.format(self.__class__.__name__)
    
dataset = KarateDataset()
data = dataset[0]

# %% [markdown]
# #### Graph Convolutional Network

# %%
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv

# GCN model with 2 layers
class Net(torch.nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = GCNConv(data.num_features, 16)
        self.conv2 = GCNConv(16, int(data.num_classes))
        
    def forward(self):
        x, edge_index = data.x, data.edge_index
        x = F.relu(self.conv1(x, edge_index))
        x = F.dropout(x, training=self.training)
        x = self.conv2(x, edge_index)
        return F.log_softmax(x, dim=1)
    
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
data = data.to(device)
model = Net().to(device)

# %% [markdown]
# #### Train the GCN model

# %%
torch.manual_seed(42)

optimizer_name = "Adam"
lr = 1e-1
optimizer = getattr(torch.optim, optimizer_name)(model.parameters(), lr=lr)
epochs = 200

def train():
    model.train()
    optimizer.zero_grad()
    loss = F.nll_loss(model()[data.train_mask], data.y[data.train_mask])
    loss.backward()
    optimizer.step()
    return loss
    
@torch.no_grad()
def test():
    model.eval()
    logits = model()
    mask1 = data['train_mask']
    pred1 = logits[mask1].max(1)[1]
    acc1 = pred1.eq(data.y[mask1]).sum().item() / mask1.sum().item()
    mask = data['test_mask']
    pred = logits[mask].max(1)[1]
    acc = pred.eq(data.y[mask]).sum().item() / mask.sum().item()
    return acc1, acc

for epoch in range(1, epochs + 1):
    loss = train()
    print(f'[{epoch}]/[{epochs}]({epoch / epochs * 100:.2f}%) \t Loss: {loss.item():.2f}')
    
train_acc, test_acc = test()
print('#' * 70)
print('Train Accuracy: %s' %train_acc )
print('Test Accuracy: %s' % test_acc)
print('#' * 70)

# %% [markdown]
# ## Part2: Using Node2Vec embeddings as input features to our GNN model

# %%
from torch_geometric.nn import Node2Vec

# reffered pytorch_geometric nodoe2Vec examples (https://github.com/pyg-team/pytorch_geometric/blob/master/examples/node2vec.py)
embedding_model = Node2Vec(data.edge_index, embedding_dim=128, walk_length=20, context_size=10, walks_per_node=10).to(device)
embedding_loader = embedding_model.loader(batch_size=128, shuffle=True)
embedding_optimizer = torch.optim.Adam(list(embedding_model.parameters()), lr=0.01)

def embedding_train():
    embedding_model.train()
    total_loss = 0
    for pos_rw, neg_rw in embedding_loader:
        embedding_optimizer.zero_grad()
        loss = embedding_model.loss(pos_rw.to(device), neg_rw.to(device))
        loss.backward()
        embedding_optimizer.step()
        total_loss += loss
    return total_loss / len(embedding_loader)

@torch.no_grad()
def embedding_test():
    embedding_model.eval()
    z = embedding_model()
    acc = embedding_model.test(z[data.train_mask], data.y[data.train_mask],
                               z[data.test_mask], data.y[data.test_mask], 
                               max_iter=150)
    return acc

for epoch in range(1, 101):
    loss = embedding_train()
    acc = embedding_test()
    print(f'Epoch: {epoch:02d}, Loss: {loss:.4f}, Acc: {acc:.4f}')

# %% [markdown]
# Visulaize embegging data in 2D space through the t-SNE

# %%
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt

color_map = ['blue' if l == 0 else 'red' for l in labels]
embeddings = embedding_model(torch.arange(data.num_nodes, device=device)).detach()

# transform the embeddings from 128 dimensions to 2D space
m = TSNE(random_state=42)
tsne_features = m.fit_transform(embeddings.cpu().numpy())

# plot the transformed embeddings
plt.figure(figsize=(9, 6))
plt.scatter(x=tsne_features[:, 0], y=tsne_features[:, 1],
            c=color_map, s=600, alpha=0.6)

# add annotations
for i, label in enumerate(np.arange(0,34)):
    plt.annotate(label, (tsne_features[:,0][i], tsne_features[:,1][i]))
    
plt.show()

# %% [markdown]
# Train the GCN model in node2vec embedding

# %%
# custom dataset
class KarateDataset(InMemoryDataset):
    def __init__(self, transform=None):
        super(KarateDataset, self).__init__('.', transform, None, None)
        
        data = Data(edge_index=edge_index)
        
        data.num_nodes = G.number_of_nodes()
        
        # embedding
        data.x = embeddings.type(torch.float32)
        
        # labels
        y = torch.from_numpy(labels).type(torch.long)
        data.y = y.clone().detach()
        
        data.num_classes = 2
        
        # splitting the data into train, validation and test
        X_train, X_test, y_train, y_test = train_test_split(pd.Series(G.nodes()), 
                                                            pd.Series(labels),
                                                            test_size=0.30,
                                                            random_state=42)
        
        n_nodes = G.number_of_nodes()
        
        # create train and test masks for data
        train_mask = torch.zeros(n_nodes, dtype=torch.bool)
        test_mask = torch.zeros(n_nodes, dtype=torch.bool)
        train_mask[X_train.index] = True
        test_mask[X_test.index] = True
        data['train_mask'] = train_mask
        data['test_mask'] = test_mask 
        
        self.data, self.slices = self.collate([data])
        
    def _download(self):
        return

    def _process(self):
        return

    def __repr__(self):
        return '{}()'.format(self.__class__.__name__)
    
dataset = KarateDataset()
data = dataset[0].to(device)

model = Net().to(device)
optimizer_name = "Adam"
lr = 1e-1
optimizer = getattr(torch.optim, optimizer_name)(model.parameters(), lr=lr)
epochs = 200

for epoch in range(1, epochs + 1):
    loss = train()
    print(f'[{epoch:3d}]/[{epochs}]({epoch / epochs * 100:.2f}%) \t Loss: {loss.item():.4f}')
    
train_acc, test_acc = test()
print('#' * 70)
print('Train Accuracy: %s' %train_acc )
print('Test Accuracy: %s' % test_acc)
print('#' * 70)
# %%
