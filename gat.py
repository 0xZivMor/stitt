import torch
import torch.nn as nn
from ogb.graphproppred.mol_encoder import AtomEncoder
from typing import Tuple, Optional, Iterable

from gat_Aleksa.GAT import *

class GatGraphClassifier(nn.Module):

    def __init__(
        self,
        d_input: int,
        d_ffn: int,
        n_heads: int,
        n_layers: int,
        n_classes: int,
        device: torch.device,
        *args,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.gat = GAT(num_of_layers=n_layers, num_heads_per_layer=[n_heads]*n_layers, num_features_per_layer=[d_ffn]*n_layers, add_skip_connection=True, bias=True,
                 dropout=0.6, layer_type=LayerType.IMP2, log_attention_weights=False)
        self.classifier = nn.Linear(d_input, n_classes)

    def forward(
        self,
        features: torch.Tensor,
        eigvects: torch.Tensor,
        attn_mask: torch.Tensor,
    ) -> torch.Tensor:

        x = self.gat([features, attn_mask])
        x = torch.mean(x, dim=1)
        
        return self.classifier(x)
    

