import torch
import torch_geometric as pyg
import scipy.sparse as sp

from typing import Tuple, Optional
from tqdm import tqdm

from torch.utils.data import Dataset

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


class SpectralDataset(Dataset):
    def __init__(self, data):
        self.data = data

    def __getitem__(self, index):
        return self.data[index]

    def __len__(self):
        return len(self.data)


def create_spectral_dataset(dataset: pyg.data.Dataset, max_graph_size: Optional[int] = None) -> Dataset:
    """
    Create a spectral dataset from a given PyTorch Geometric dataset. The dataset contains the node features and the graph's Laplacian eigenvector padded with zeros so all graphs in the dataset contain the same number of nodes. Attention mask can be used to differ between true and padded data.
    The created dataset ignores edges features.

    Args:
        dataset (pyg.data.Dataset): The input PyTorch Geometric dataset.
        max_graph_size (Optional[int]): The maximum size of the graphs in the dataset. If not provided, it will be
            determined as the maximum number of nodes among all graphs in the dataset.

    Returns:
        Dataset: The created spectral dataset in the following format:
        (padded graph nodes, padded eigenvectors, attention mask, label)

    """

    # Determine the maximum graph size if not provided
    if max_graph_size is None:
        max_graph_size = max([graph.num_nodes for graph in dataset])

    data = []
    for graph in tqdm(iter(dataset), total=len(dataset), desc="Creating Spectral Dataset"):
        # Compute the Laplacian eigenvectors for the graph
        _, eigenvects = get_laplacian_eig(graph)

        # Pad the eigenvectors to match the maximum graph size
        padded_eigenvects = torch.zeros(max_graph_size, max_graph_size)
        padded_eigenvects[: eigenvects.size(0), : eigenvects.size(1)] = eigenvects

        # Pad the node features to match the maximum graph size
        node_features = graph.x
        padding = torch.zeros(max_graph_size - graph.num_nodes, node_features.size(1))
        node_features = torch.cat([graph.x, padding], dim=0)

        # Get the label and create the attention mask
        label = graph.y
        attention_mask = torch.zeros(max_graph_size)
        attention_mask[:graph.num_nodes] = 1

        # Append the data tuple to the list
        data.append((node_features, padded_eigenvects, attention_mask, label))

    return SpectralDataset(data)
