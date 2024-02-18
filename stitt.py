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


class AttentionHead(nn.Module):
    """
    This class represents an attention head for transformer models.
    """

    def __init__(self, d_input: int, n_hidden: int):
        """
        Initializes the AttentionHead.

        Args:
            d_input: the dimension of the input
            n_hidden: the dimension of the keys, queries, and values
        """
        super().__init__()
        self.W_K = nn.Linear(d_input, n_hidden)
        self.W_Q = nn.Linear(d_input, n_hidden)
        self.W_V = nn.Linear(d_input, n_hidden)
        self.n_hidden = n_hidden

    def forward(
        self, x: torch.Tensor, attn_mask: Optional[torch.Tensor]
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Computes the forward pass of the attention head.

        Args:
            x (torch.Tensor): The input tensor. Shape: (batch_size, seq_length, d_input)
            attn_mask (Optional[torch.Tensor]): The causal mask tensor. If provided, it acts as an attention mask
            that determines which tokens in the sequence should be attended to. It's a 3D tensor where the value at
            position [b, i, j] is 1 if the token at position i in batch b should attend to the token at position j,
            and 0 otherwise. If not provided (None), ignore it.
            Shape: (batch_size, seq_length, seq_length)

        Returns:
            attn_output (torch.Tensor): The output tensor after attention. Shape: (batch_size, seq_length, n_hidden)
            attn_score (torch.Tensor): The attention score tensor. Shape: (batch_size, seq_length, seq_length)
        """
        # Assuming all inputs are of the correct shape, not verifying

        # Compute the keys, queries, and values
        K = self.W_K(x)
        Q = self.W_Q(x)
        V = self.W_V(x)

        # Compute the attention scores
        attention_scores = Q @ K.transpose(-2, -1) / np.sqrt(self.n_hidden)

        # Compute masked attention scores, if mask is provided
        if attn_mask is not None:
            masked_attention = torch.where(attn_mask == 1, attention_scores, -torch.inf)
            # masked_attention = attention_scores * attn_mask
            attn_score = nn.functional.softmax(masked_attention, dim=-1)
        else:
            attn_score = nn.functional.softmax(attention_scores, dim=-1)

        attn_output = attn_score @ V

        return attn_output, attn_score


class MultiheadAttention(nn.Module):
    def __init__(self, d_input: int, n_hidden: int, num_heads: int):
        """
        Initializes the MultiheadAttention.

        Args:
            d_input (int): The dimension of the input.
            n_hidden: the hidden dimenstion for the attention layer
            num_heads (int): The number of attention heads.
        Attributes:
            attention_heads (nn.ModuleList): A list of attention heads.
            W_proj (nn.Linear): A linear layer for projecting the concatenated outputs of the attention heads back
            to the original dimension.
        """

        super().__init__()

        self.d_input = d_input
        self.n_hidden = n_hidden
        self.num_heads = num_heads

        self.attention_heads = nn.ModuleList(
            [AttentionHead(d_input, n_hidden) for _ in range(num_heads)]
        )
        self.W_proj = nn.Linear(n_hidden * num_heads, d_input)

    def forward(
        self, x: torch.Tensor, attn_mask: Optional[torch.Tensor]
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Executes the forward pass of the multi-head attention mechanism.

        Args:
            x (torch.Tensor): The input tensor. It has a shape of (batch_size, seq_length, d_input).
            attn_mask (Optional[torch.Tensor]): The attention mask tensor. If provided, it serves as an attention guide
            that specifies which tokens in the sequence should be attended to. It's a 3D tensor where the value at
            position [b, i, j] is 1 if the token at position i in batch b should attend to the token at position j,
            and 0 otherwise. If not provided (None), ignore it.
            Shape: (batch_size, seq_length, seq_length)

        Returns:
            attn_output (torch.Tensor): The output tensor after applying multi-head attention. It has a shape of
            (batch_size, seq_length, d_input).

        This method computes the multi-head attention by looping through each attention head, collecting the outputs,
        concatenating them together along the hidden dimension, and then projecting them back into the output dimension
        (d_input). It returns both the final attention outputs as well as the attn_scores from each head.
        """
        attn_output, attn_scores = None, None

        # Assuming all inputs are of the correct shape, not verifying

        scores = []
        attns = []

        # Sequentially apply each attention head
        for i, head in enumerate(self.attention_heads):
            head_output, head_scores = head(
                x, attn_mask
            )  # if attn_mask is None, it will be ignored
            scores.append(head_scores)
            attns.append(head_output)

        # Project the concatenated outputs back to the original dimension
        attns = torch.cat(attns, dim=-1)
        attn_output = self.W_proj(attn_output)

        attn_scores = torch.stack(scores, dim=1)

        return attn_output, attn_scores


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
        self.attention = MultiheadAttention(d_input, attn_dim, num_heads)
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

        attended_values, attention_weights = self.self_attention(x)
        normalized1 = self.rmsnorm1(attended_values)
        residual1 = x + normalized1

        ffn = self.ffn(residual1)
        normalized2 = self.rmsnorm2(ffn)
        residual2 = normalized1 + normalized2

        if self.return_attention:
            return residual2, attention_weights
        else:
            return residual2
