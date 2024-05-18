import torch
import torch.nn as nn
from ogb.graphproppred.mol_encoder import AtomEncoder
from typing import Tuple, Optional, Iterable
import torch.nn.functional as F
from torch.nn import Linear, Dropout
from torch_geometric.nn import GATConv, GATv2Conv
from torch_geometric.nn import GATConv

class GatGraphClassifier(nn.Module):

    def __init__(
        self,
        d_input: int,
        n_heads: int,
        n_layers: int,
        n_classes: int,
        d_atom_emb: int,
        device: torch.device,
        *args,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.atom_features_encoder = AtomEncoder(emb_dim=d_atom_emb)
        self.gat = GAT(d_input, n_layers,n_heads)
        self.classifier = nn.Linear(d_input, n_classes)

    def forward(
        self,
        nodes_features: torch.Tensor,
        edges_connectivity: torch.Tensor,
    ) -> torch.Tensor:
        
        embeded_nodes_features = torch.stack([self.atom_features_encoder(atom_features) for atom_features in nodes_features], dim=0)
        x = self.gat(embeded_nodes_features, edges_connectivity)
        x = torch.mean(x, dim=1)
        
        return self.classifier(x)

    
class GAT(torch.nn.Module):

    def __init__(self,d_input, n_layers, n_heads, dropout=0.6, activiation = nn.GELU()):
        super().__init__()
        self.d_input = d_input
        self.n_layers = n_layers
        self.n_heads = n_heads
        self.dropout = dropout
        self.activiation = activiation

        #side note, self loops is True and if I remember correctly, self loops creates skip connections. 
        self.gat_layers = [
            GATConv(in_channels=d_input, out_channels=d_input, heads=n_heads, dropout=dropout)
            for _ in range(n_layers)
        ]

    def forward(self,embeded_nodes_features, edges_connectivity):
        # expects nodes_features shape [Batch, V, embeded_nodes_features]
        # expects edges_connectivity shape [Batch, V]

        #return shape [Batch, V, embeded_nodes_features]
        
        for i, gat_layer in enumerate(self.gat_layers):
            F.dropout(embeded_nodes_features, p=self.dropout, inplace=True)
            embeded_nodes_features = gat_layer(embeded_nodes_features,edges_connectivity)
            if i < self.n_layers - 1:
                embeded_nodes_features = self.activiation(embeded_nodes_features)

        return embeded_nodes_features