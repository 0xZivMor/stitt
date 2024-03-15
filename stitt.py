import torch
import torch.nn as nn
from ogb.graphproppred.mol_encoder import AtomEncoder

from typing import Tuple, Optional, Iterable


class RMSNorm(nn.Module):
    def __init__(self, feature_dim: int, eps=1e-8):
        super().__init__()
        self.rms = nn.Parameter(torch.ones(feature_dim) * 0.1)
        self.eps = eps

    def forward(self, x):
        return x / torch.sqrt(
            torch.mean(self.rms * x**2, dim=-1, keepdim=True) + self.eps
        )

class FFN(nn.Module):
    """
    This class represents a Feed-Forward Network (FFN) layer.
    """

    def __init__(self, d_input: int, n_hidden: int, d_out: Optional[int] = None):
        """
        Args:
            d_input (int): The dimension of the input.
            n_hidden (int): The hidden dimension for the FFN.
        Attributes:
            ffn (nn.Sequential): A sequential container of the FFN layers.
        """

        super().__init__()
        if d_out is None:
            d_out = d_input

        self.ffn = nn.Sequential(
            nn.Linear(d_input, n_hidden),
            nn.GELU(),
            nn.Linear(n_hidden, d_out),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Executes the forward pass of the FFN.

        Args:
            x (torch.Tensor): The input tensor. Shape: (batch_size, seq_length, d_input)

        Returns:
            out (torch.Tensor): The output tensor of the FFN. Shape: (batch_size, seq_length, d_input)
        """
        return self.ffn(x)


class TransformerBlock(nn.Module):
    """
    This class represents a Transformer block.

    Args:
      d_input (int): The dimension of the input.
      attn_dim (int): The hidden dimension for the attention layer.
      mlp_dim (int): The hidden dimension for the FFN.
      num_heads (int): The number of attention heads.

    Attributes:
      rmsnomr1: The first layer normalization layer for the attention.
      rmsnorm2: The second layer normalization layer for the FFN.
      attention (MultiheadAttention): The multi-head attention mechanism.
      ffn (FFN): The feed-forward network.
    """

    def __init__(self, d_input: int, attn_dim: int, mlp_dim: int, num_heads: int, return_attention: Optional[bool]=True):
        super().__init__()
        self.d_input = d_input
        self.attn_dim = attn_dim
        self.mlp_dim = mlp_dim
        self.num_heads = num_heads
        self.return_attention = return_attention

        self.rmsnorm1 = RMSNorm(attn_dim)
        self.rmsnorm2 = RMSNorm(d_input)
        self.attention = nn.MultiheadAttention(
            attn_dim,
            num_heads,
            batch_first=True,
            kdim=d_input,
            vdim=d_input,
        )
        self.ffn = FFN(d_input, mlp_dim)

    def forward(self, x: torch.Tensor, attn_mask: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Executes the forward pass of the Transformer block with pre RMSnorm.

        Args:
          x (torch.Tensor): The input tensor. Shape: (batch_size, seq_length, d_input)

        Returns:
          x (torch.Tensor): The output tensor after passing through the Transformer block. Shape: (batch_size, seq_length, d_input)
          attn_scores (torch.Tensor): The attention weights of each of the attention heads. Shape: (batch_size, num_heads, seq_length, seq_length)
        """

        attended_values, attention_weights = self.attention(x, x, x, attn_mask=attn_mask)
        normalized1 = self.rmsnorm1(attended_values)
        residual1 = x + normalized1

        ffn = self.ffn(residual1)
        normalized2 = self.rmsnorm2(ffn)
        residual2 = normalized1 + normalized2

        if self.return_attention:
            return residual2, attention_weights
        else:
            return residual2


class Stitt(nn.Module):
    """
    Stitt module for graph processing using transformer.

    Args:
      d_input (int): Dimension of the input embeddings.
      d_attn (int): Dimension of the attention embeddings. Also, maximum size of input graph.
      d_ffn (int): Dimension of the feed-forward network embeddings.
      num_heads (int): Number of attention heads.
      n_layers (int): Number of transformer blocks.
      node_features (Optional[int]): Number of input node features. Default is 0.

    Attributes:
      n_features (int): Number of node features.
      d_input (int): Dimension of the input embeddings.
      features_embed (AtomEncoder): Linear layer for embedding node features.
      geometric_embed (nn.Linear): Linear layer for embedding geometric features.
      transformer_blocks (nn.ModuleList): List of transformer blocks.

    Returns:
      torch.Tensor: Output tensor after passing through the Stitt module.

    Raises:
      ValueError: If the number of graphs and eigenvectors in the batch are not the same.


    """

    def __init__(
        self,
        d_input: int,
        d_attn: int,
        d_ffn: int,
        max_graph: int,
        num_heads: int,
        n_layers: int,
        device: torch.device,
        node_features: Optional[int] = 0,
        *args,
        **kwargs,
    ):
        super(Stitt, self).__init__(*args, **kwargs)

        self.n_features = node_features
        self.d_input = d_input
        self._device = device
        self.max_graph = max_graph  # the maximum number of nodes in a graph the model will handle
        self.num_heads = num_heads

        self.features_embed = AtomEncoder(emb_dim=d_input)
        self.geometric_embed = nn.Linear(self.max_graph, d_input)

        self.transformer_blocks = nn.ModuleList(
            [
                TransformerBlock(d_input, d_attn, d_ffn, num_heads, False)
                for _ in range(n_layers)
            ]
        )

    def forward(self, features: torch.Tensor, eigvects: torch.Tensor, attn_mask: torch.Tensor) -> torch.Tensor:
        # expects features shape [Batch, V, features]
        # expects eigvects shape [Batch, V, V]
        # expects attn_mask shape [Batch, V]

        embedded_features = []
        
        # embedding can't be don't parallely for some reason
        for datum in features:
            embedded_features.append(self.features_embed(datum))
        embedded_features = torch.stack(embedded_features, dim=0)
        
        # pad the eigenvectors to fit the expected shape
        b, n, _ = eigvects.shape
        padded_eigvects = torch.zeros((b, n, self.max_graph))
        padded_eigvects[:, :, :n] = eigvects

        embedded_geometrics = self.geometric_embed(padded_eigvects)

        x = embedded_features + embedded_geometrics
        
        # make a copy of the attention mask for each head
        attn_mask = torch.cat([attn_mask] * self.num_heads)

        for block in self.transformer_blocks:
            x = block(x, attn_mask)
        return x


class StittGraphClassifier(nn.Module):

    def __init__(
        self,
        d_input: int,
        d_attn: int,
        n_heads: int,
        d_ffn: int,
        n_layers: int,
        n_classes: int,
        *args,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.stitt = Stitt(d_input, d_attn, d_ffn, n_heads, n_layers)
        self.classifier = nn.Linear(d_input, n_classes)

    def forward(self, features: torch.Tensor, eigvects: torch.Tensor) -> torch.Tensor:

        x = self.stitt(features, eigvects)
        return self.classifier(x)