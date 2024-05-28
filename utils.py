import torch
import torch_geometric as pyg
import scipy.sparse as sp
import numpy as np

from typing import Optional, Iterable, Tuple
from tqdm import tqdm

from ogb.graphproppred import Evaluator

from torch.utils.data import Dataset
from torch_geometric.loader import DataLoader


from torch.nn.utils.rnn import pad_sequence


from sys import getsizeof as gso


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

def pad_graph(node_features: torch.Tensor, eigenvects: torch.Tensor, max_graph: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Pads the node features and eigenvectors of a graph with zeros to match a maximum graph size.

    Args:
        node_features (torch.Tensor): The node features of the graph.
        eigenvects (torch.Tensor): The eigenvectors of the graph.
        max_graph (int): The maximum size of the graph.

    Returns:
        tuple: A tuple containing the padded node features, padded eigenvectors, and attention mask.
    """
    
    num_nodes = node_features.size(0)
    
    # Create tensors to store the padded node features and eigenvectors
    padded_node_features = torch.zeros((max_graph, node_features.size(1)), dtype=node_features.dtype)
    padded_eigenvects = torch.zeros((max_graph, max_graph), dtype=eigenvects.dtype)
    
    # Copy the original node features and eigenvectors to the padded tensors
    padded_eigenvects[:num_nodes, :num_nodes] = eigenvects
    padded_node_features[:num_nodes, :] = node_features
    
    # Create an attention mask to indicate the valid nodes in the graph
    attention_mask = torch.zeros((max_graph, max_graph))
    attention_mask[:num_nodes, :num_nodes] = 1
    
    return padded_node_features, padded_eigenvects, attention_mask


def create_spectral_dataset(dataset: pyg.data.Dataset, upsample: Optional[Iterable[int]] = None) -> Dataset:
    """
    Create a spectral dataset from a given PyTorch Geometric dataset. The dataset contains the node features and the graph's Laplacian eigenvector padded with zeros so all graphs in the dataset contain the same number of nodes. Attention mask can be used to differ between true and padded data.
    The created dataset ignores edges features.
    IMPORTANT: When the dataset wrapped with a DataLoader, use the 
    collate_spectral_dataset function as collate_fn. Data in the dataset
    are not of the same size and require some processing (per batch).

    Args:
        dataset (pyg.data.Dataset): The input PyTorch Geometric dataset.

        upsample (Optional[Iterable[int]], optional): An optional iterable specifying the number of times each graph should be upsampled. If provided, the function will create additional samples by permuting the graphs in the dataset. Defaults to None.

    Returns:
        Dataset: The created spectral dataset in the following format:
        (padded graph nodes, padded eigenvectors, attention mask, label)

    """
    data = []

    # Iterate over each graph in the dataset
    for graph in tqdm(iter(dataset), total=len(dataset), desc="Creating Spectral Dataset"):

        _, eigenvects = get_laplacian_eig(graph)
        
        if not upsample or not all(upsample):
            data.append((graph.x, eigenvects, graph.y, graph.num_nodes))
        else:
            # Upsample the graph based on the specified number of times
            for _ in range(upsample[graph.y]):
                permuted_graph, perm = permute_graph(graph, True)
                data.append((permuted_graph.x, eigenvects[perm, :], graph.y, graph.num_nodes))

    return SpectralDataset(data)

def create_spectral_dataset2(dataset: pyg.data.Dataset, upsample: Optional[Iterable[int]] = None) -> Dataset:
    """
    Create a spectral dataset from a given PyTorch Geometric dataset. The dataset contains the node features and the graph's Laplacian eigenvector padded with zeros so all graphs in the dataset contain the same number of nodes. Attention mask can be used to differ between true and padded data.
    The created dataset ignores edges features.
    IMPORTANT: When the dataset wrapped with a DataLoader, use the 
    collate_spectral_dataset function as collate_fn. Data in the dataset
    are not of the same size and require some processing (per batch).

    Args:
        dataset (pyg.data.Dataset): The input PyTorch Geometric dataset.

        upsample (Optional[Iterable[int]], optional): An optional iterable specifying the number of times each graph should be upsampled. If provided, the function will create additional samples by permuting the graphs in the dataset. Defaults to None.

    Returns:
        Dataset: The created spectral dataset in the following format:
        (padded graph nodes, padded eigenvectors, attention mask, label)

    """
    max_graph = max([g.num_nodes for g in dataset])
    data = []

    # Iterate over each graph in the dataset
    iterable = tqdm(iter(dataset), total=len(dataset), desc="Creating Spectral Dataset:")
    for graph in iterable:
        if not upsample:
            _, eigenvects = get_laplacian_eig(graph)
            padded_features, padded_eigenvects, attention_mask = pad_graph(graph.x, eigenvects, max_graph)
            data.append((padded_features, padded_eigenvects, attention_mask, graph.y))
        else:
            # Upsample the graph based on the specified number of times
            for _ in range(upsample[graph.y]):
                permuted_graph = permute_graph(graph)
                _, eigenvects = get_laplacian_eig(permuted_graph)
                padded_features, padded_eigenvects, attention_mask = pad_graph(graph.x, eigenvects, max_graph)
                data.append((padded_features, padded_eigenvects, attention_mask, graph.y))
        iterable.set_postfix({"Memory Size": gso(data)})


    return SpectralDataset(data)

def collate_spectral_dataset(batch):
    """
    Collates a batch of spectral datasets.

    Args:
        batch (list): A list of tuples containing the sequences, eigenvectors, labels, and number of nodes for each sample.

    Returns:
        tuple: A tuple containing the padded node features, padded eigenvectors, attention masks, and concatenated labels.
    """

    # Separate the sequences, labels, and attention masks
    node_features, eigenvectors, labels, num_nodes = zip(*batch)
    batch_size = len(batch)

    max_graph = max(num_nodes)

    # Pad the sequences with zeros
    padded_node_features = pad_sequence(node_features, batch_first=True, padding_value=0)

    padded_eigenvectors = torch.zeros((batch_size, max_graph, max_graph))
    attention_masks = torch.zeros((batch_size,
                                   max_graph,
                                   max_graph))

    # pad eigenvectors for batching and construct attention mask accordingly
    for i, ev_size in enumerate(zip(eigenvectors, num_nodes)):
        ev, size = ev_size
        padded_eigenvectors[i, :size, :size] = ev
        attention_masks[i, :size, :size] = 1

    return (padded_node_features, padded_eigenvectors, 
            attention_masks, torch.concat(labels).flatten())

def collate_spectral_dataset_no_eigenvects(batch):
    """
    Collates a spectral dataset without eigenvectors.

    Args:
        batch (list): A list of samples in the dataset.

    Returns:
        tuple: A tuple containing the collated dataset with zeroed eigenvectors, essentially providing no geometric information.
    """
    
    collated = collate_spectral_dataset(batch)
    
    # Return the collated dataset without eigenvectors
    return (collated[0], torch.zeros_like(collated[1]), collated[2], collated[3])

def collate_dataset_for_gat(batch):
    """
    Args:
        batch (list): A list of samples in the dataset.

    Returns:
        tuple: A tuple containing the collated dataset for gat
    """
    
    collated = collate_spectral_dataset(batch)
    
    # Return the collated dataset without eigenvectors
    return (collated[0], collated[2], collated[3])


def permute_graph(graph: pyg.data.Data, return_perm: Optional[bool]=True) -> pyg.data.Data:
    """
    Permutes the given graph by randomly shuffling the node features and edge indices.

    Args:
        graph (pyg.data.Data): The input graph to be permuted.
        return_perm (bool, optional): Whether to return the permutation indices. 
                                      Defaults to True.

    Returns:
        pyg.data.Data: The permuted graph with shuffled node features and edge indices.
        Optional[torch.Tensor]: The permutation indices if `return_perm` is True, else None.
    """

    perm = torch.randperm(graph.num_nodes)
    
    # Permute the node features
    permuted_x = graph.x[perm]
    
    # Permute the edge indices
    permuted_edge_index = permute_edge_index(graph.edge_index, perm)
    
    # Create the permuted graph
    permuted_graph = pyg.data.Data(x=permuted_x, y=graph.y, edge_index=permuted_edge_index)
    
    if return_perm:
        return permuted_graph, perm
    else:
        return permuted_graph

def permute_edge_index(edge_index: torch.Tensor, perm: torch.Tensor) -> torch.Tensor:
    """
    Permutes the edge indices of a given graph.

    Args:
        edge_index (torch.Tensor): The edge indices of the graph.
        perm (torch.Tensor): The permutation indices.

    Returns:
        torch.Tensor: The permuted edge indices.
    """
    # Permute the row indices
    permuted_row = perm[edge_index[0]]
    # Permute the column indices
    permuted_col = perm[edge_index[1]]
    
    # Create the permuted edge indices
    permuted_edge_index = torch.stack([permuted_row, permuted_col], dim=0)
    return permuted_edge_index

def evaluate_model(model: torch.nn.Module, val_loader: Iterable, batch_size: Optional[int]=32, no_eigenvects: Optional[bool] = False):

    if torch.cuda.is_available():
        device = torch.device("cuda")

    if torch.backends.mps.is_available():
        # device = torch.device("mps")
        # print("Using MPS")
        device = torch.device("cpu")
        print("Using CPU")

    # if isinstance(dataset, Dataset):
    #     val_loader = DataLoader(
    #         dataset, batch_size=batch_size, collate_fn=collate_spectral_dataset
    #     )
    # elif isinstance(dataset, DataLoader):
    #     val_loader = dataset
    # else:
    #     raise ValueError("dataset must be an instance of Dataset or DataLoader")
    
    evaluator = Evaluator(name="ogbg-molhiv")
    y_true = []
    y_pred = []

    model.eval()

    with torch.no_grad():
        for data in tqdm(val_loader):

            outputs = model(data.x, data.edge_index, data.batch)
            predicted = torch.argmax(outputs, dim=1)

            y_true.append(data.y.flatten())
            y_pred.append(predicted)

    y_true = torch.concat(y_true).unsqueeze(1)
    y_pred = torch.concat(y_pred).unsqueeze(1)

    roc_auc = evaluator.eval({"y_true": y_true, "y_pred": y_pred})["rocauc"]
    print(f"ROC AUC: {roc_auc}")
    return roc_auc

