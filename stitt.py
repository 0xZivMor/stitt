import torch
import torch.nn as nn
import torch_geometric as pyg
import scipy.sparse as sp

from typing import Tuple, Optional, Iterable


def get_laplacian_eig(
    graph: pyg.data.Data, normalization: Optional[str] = None
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Compute the eigenvalues and eigenvectors of the normalized Laplacian matrix of a graph.

    Args:
      graph (pyg.data.Data): The input graph.
      normalization (Optional[str]): The type of normalization to be applied to the Laplacian matrix. Default is None.

    Returns:
      tuple: A tuple containing the sorted eigenvalues and eigenvectors of the normalized Laplacian matrix.
    """

    laplacian = pyg.utils.get_laplacian(graph.edge_index, normalization=normalization)
    row, col = laplacian[0]
    sparse_lap = sp.coo_matrix(
        (laplacian[1], (row, col)), (graph.num_nodes, graph.num_nodes)
    )

    # Eigendecomposition on sparse matrices is not complete
    lap = torch.tensor(sparse_lap.toarray())

    # Compute the eigendecomposition of the Laplacian
    eigenvalues, eigenvectors = torch.linalg.eigh(lap)
    eigenvalues = torch.real(eigenvalues)
    eigenvectors = torch.real(eigenvectors)

    # Sort the eigenvalues and eigenvectors in descending order
    sorted_indices = torch.argsort(eigenvalues, descending=True)
    eigenvalues = eigenvalues[sorted_indices]
    eigenvectors = eigenvectors[:, sorted_indices]

    return eigenvalues, eigenvectors


class RMSNorm(nn.Module):
    def __init__(self, feature_dim: int, eps=1e-8):
        super().__init__()
        self.rms = nn.Parameter(torch.ones(feature_dim) * 0.1)
        self.eps = eps

    def forward(self, x):
        return x / torch.sqrt(
            torch.mean(self.rms * x**2, dim=-1, keepdim=True) + self.eps
        )


# FFN class is alreay implemented for you
class FFN(nn.Module):
    """
    This class represents a Feed-Forward Network (FFN) layer.
    """

    def __init__(self, d_input: int, n_hidden: int):
        """
        Args:
            d_input (int): The dimension of the input.
            n_hidden (int): The hidden dimension for the FFN.
        Attributes:
            ffn (nn.Sequential): A sequential container of the FFN layers.
        """

        super().__init__()
        self.ffn = nn.Sequential(
            nn.Linear(d_input, n_hidden),
            nn.GELU(),
            nn.Linear(n_hidden, d_input),
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
    """

    def __init__(self, d_input: int, attn_dim: int, mlp_dim: int, num_heads: int):
        """
        Args:
            d_input (int): The dimension of the input.
            attn_dim (int): The hidden dimension for the attention layer.
            mlp_dim (int): The hidden dimension for the FFN.
            num_heads (int): The number of attention heads.

        Attributes:
            layer_norm_1 (nn.LayerNorm): The first layer normalization layer for the attention.
            layer_norm_2 (nn.LayerNorm): The second layer normalization layer for the FFN.
            attention (MultiheadAttention): The multi-head attention mechanism.
            ffn (FFN): The feed-forward network.
        """
        super().__init__()
        self.d_input = d_input
        self.attn_dim = attn_dim
        self.mlp_dim = mlp_dim
        self.num_heads = num_heads

        self.rmsnorm1 = RMSNorm(attn_dim)
        self.rmsnorm2 = RMSNorm(mlp_dim)
        self.attention = nn.MultiheadAttention(attn_dim, num_heads, batch_first=True)
        self.ffn = FFN(d_input, mlp_dim)

    def forward(
        self, x: torch.Tensor, attn_mask: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Executes the forward pass of the Transformer block with pre RMSnorm.

        Args:
            x (torch.Tensor): The input tensor. Shape: (batch_size, seq_length, d_input)
            attn_mask (torch.Tensor): The attention mask tensor. If provided, it serves as an attention guide
            that specifies which tokens in the sequence should be attended to. It's a 3D tensor where the value at
            position [b, i, j] is 1 if the token at position i in batch b should attend to the token at position j,
            and 0 otherwise. If not provided (None), no specific attention pattern is enforced.
            Shape: (batch_size, seq_length, seq_length)

        Returns:
            x (torch.Tensor): The output tensor after passing through the Transformer block. Shape: (batch_size, seq_length, d_input)
            attn_scores (torch.Tensor): The attention weights of each of the attention heads. Shape: (batch_size, num_heads, seq_length, seq_length)
        """

        attended_values, attention_weights = self.self_attention(x, x, x)
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
  Stitt module for graph processing.

  Args:
    node_features (int): Number of input node features.
    d_input (int): Dimension of the input embeddings.
    d_attn (int): Dimension of the attention embeddings.
    d_ffn (int): Dimension of the feed-forward network embeddings.
    num_heads (int): Number of attention heads.
    n_layers (int): Number of transformer blocks.

  Attributes:
    n_features (int): Number of node features.
    d_input (int): Dimension of the input embeddings.
    features_embed (nn.Linear): Linear layer for embedding node features.
    geometric_embed (nn.Linear): Linear layer for embedding geometric features.
    transformer_blocks (nn.ModuleList): List of transformer blocks.

  """

  def __init__(self, node_features, d_input, d_attn, d_ffn, num_heads, n_layers):
    super(Stitt, self).__init__()

    self.n_features = node_features
    self.d_input = d_input

    self.features_embed = nn.Linear(node_features, d_input)
    self.geometric_embed = nn.Linear(d_input, d_input)

    self.transformer_blocks = nn.ModuleList(
      [
        TransformerBlock(d_input, d_attn, d_ffn, num_heads)
        for _ in range(n_layers)
      ]
    )

  def forward(self, features_batch: torch.Tensor, eigvects_batch=torch.Tensor) -> torch.Tensor:
    """
    Forward pass of the Stitt module.

    Args:
      features_batch (torch.Tensor): Batch of input node features.
      eigvects_batch (torch.Tensor): Batch of input eigenvectors.

    Returns:
      torch.Tensor: Output tensor after passing through the Stitt module.

    Raises:
      ValueError: If the number of graphs and eigenvectors in the batch are not the same.

    """

    # batches should have the same size
    if features_batch.size(0) != eigvects_batch.size(0):
      raise ValueError("Must have the same number of graphs and eigenvectors")
    
    embedded_features = self.features_embed(features_batch)

    # Pad eigvects_batch with zeros up to the maximum graph size
    padded = torch.zeros((eigvects_batch.size(0),
               eigvects_batch.size(1),
               self.d_input, self.d_input))
    padded[:, :, :eigvects_batch.size(2), :eigvects_batch.size(3)] = eigvects_batch
    embedded_geometrics = self.geometric_embed(padded.detach())

    x = embedded_features + embedded_geometrics

    for block in self.transformer_blocks:
      x = block()
    return x
