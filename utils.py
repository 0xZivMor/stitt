import torch
import torch_geometric as pyg
import scipy.sparse as sp

from typing import Tuple, Optional
from tqdm import tqdm

from torch.utils.data import Dataset
from torch.nn.utils.rnn import pad_sequence

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


def create_spectral_dataset(dataset: pyg.data.Dataset) -> Dataset:
    """
    Create a spectral dataset from a given PyTorch Geometric dataset. The dataset contains the node features and the graph's Laplacian eigenvector padded with zeros so all graphs in the dataset contain the same number of nodes. Attention mask can be used to differ between true and padded data.
    The created dataset ignores edges features.
    IMPORTANT: When the dataset wrapped with a DataLoader, use the 
    collate_spectral_dataset function as collate_fn. Data in the dataset
    are not of the same size and require some processing (per batch).

    Args:
        dataset (pyg.data.Dataset): The input PyTorch Geometric dataset.

    Returns:
        Dataset: The created spectral dataset in the following format:
        (padded graph nodes, padded eigenvectors, attention mask, label)

    """
    data = []
    
    for graph in tqdm(iter(dataset), total=len(dataset), desc="Creating Spectral Dataset"):
        # Compute the Laplacian eigenvectors for the graph
        _, eigenvects = get_laplacian_eig(graph)

        data.append((graph.x, eigenvects, graph.y, graph.num_nodes))

    return SpectralDataset(data)

def collate_spectral_dataset(batch):
    
    # Separate the sequences, labels, and attention masks
    node_features, eigenvectors, labels, num_nodes = zip(*batch)
    
    max_graph = max(num_nodes)
    
    # Pad the sequences with zeros
    padded_node_features = pad_sequence(node_features, batch_first=True, padding_value=0)
    
    # Pad the eigenvectors with zeros
    padded_eigenvectors = torch.zeros((len(batch), max_graph, max_graph))
    for i, ev in enumerate(eigenvectors):
        padded_eigenvectors[i, :ev.size(0), :ev.size(1)] = ev
    
    # Pad the attention masks with zeros
    attention_masks = torch.zeros((padded_node_features.size(0),
                                   padded_node_features.size(1))).to(dtype=int)
    for i, count in enumerate(num_nodes):
        attention_masks[i, :count] = 1

    return padded_node_features, padded_eigenvectors, attention_masks, torch.concat(labels)

