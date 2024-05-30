import torch
import torch.nn as nn
from ogb.graphproppred.mol_encoder import AtomEncoder
from torch_geometric.nn import  GATv2Conv, global_mean_pool

class GatGraphClassifier(nn.Module):

    def __init__(
        self,
        d_hidden: int,
        n_heads: int,
        n_layers: int,
        n_classes: int,
        device: torch.device,
        *args,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        torch.manual_seed(42)
        self.atom_features_encoder = AtomEncoder(emb_dim=d_hidden).to(device)
        self.gat = GAT(d_hidden, n_layers,n_heads,device).to(device)
        self.classifier = nn.Linear(d_hidden, n_classes).to(device)

    def forward(
        self,
        nodes_features: torch.Tensor,
        edges_connectivity: torch.Tensor,
        batch: torch.Tensor, # contains the information of splitting the batch back to original graphs
    ) -> torch.Tensor:
        
        # expects nodes_features shape [sum of nodes in batch , nodes_features]
        # expects edges_connectivity shape [2,sum of edges in batch]
        #return shape [Batch, n_classes]

        # 1. Obtain nodes meaningful embeddings 
        x = self.atom_features_encoder(nodes_features)
        x = self.gat(x, edges_connectivity)

        # 2. Readout layer
        x = global_mean_pool(x, batch)  # [batch_size, hidden_channels]

        # 3. Apply a final classifier
        result = self.classifier(x) #in binary label case - [[prob of 0 label,prob of 1 label] for graph in batch]
        return result

    
class GAT(torch.nn.Module):

    def __init__(self,d_hidden, n_layers, n_heads,device, dropout=0, activiation = nn.GELU()):
        super().__init__()
        self.d_hidden = d_hidden
        self.n_layers = n_layers
        self.n_heads = n_heads
        self.dropout = dropout
        self.activiation = activiation


        #side note, self loops is True and if I remember correctly, self loops creates skip connections. 
        self.gat_layers = [
            GATv2Conv(in_channels=d_hidden, out_channels=d_hidden, heads=n_heads, dropout=dropout, concat = False).to(device)
            for i in range(n_layers)
        ]

    def forward(self,embeded_nodes_features, edges_connectivity):
        # expects nodes_features shape [sum of nodes in batch , nodes_features]
        # expects edges_connectivity shape [2,sum of edges in batch]

        #return shape [sum of nodes in batch , d_hidden]
        
        for i, gat_layer in enumerate(self.gat_layers):
            nn.functional.dropout(embeded_nodes_features, p=self.dropout, inplace=True, training=self.training)
            embeded_nodes_features = gat_layer(embeded_nodes_features,edges_connectivity)
            if i < self.n_layers - 1:
                embeded_nodes_features = self.activiation(embeded_nodes_features)

        return embeded_nodes_features