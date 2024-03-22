import torch
import torch_geometric as pyg
import scipy.sparse as sp

from typing import Optional, Iterable, Tuple
from tqdm import tqdm

from ogb.graphproppred import Evaluator

from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence

from torch.utils.tensorboard import SummaryWriter

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

        if not upsample or not all(upsample):
            _, eigenvects = get_laplacian_eig(graph)
            data.append((graph.x, eigenvects, graph.y, graph.num_nodes))
        else:
            # Upsample the graph based on the specified number of times
            for _ in range(upsample[graph.y]):
                permuted_graph = permute_graph(graph)
                _, eigenvects = get_laplacian_eig(permuted_graph)
                data.append((permuted_graph.x, eigenvects, graph.y, graph.num_nodes))

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

    # pad eigenvectors for batching
    padded_eigenvectors = torch.zeros((batch_size, max_graph, max_graph))
    for i, ev in enumerate(eigenvectors):
        padded_eigenvectors[i, :ev.size(0), :ev.size(1)] = ev

    # Define attention mask w.r.t padding
    attention_masks = torch.zeros((batch_size,
                                   max_graph,
                                   max_graph))
    for i, count in enumerate(num_nodes):
        attention_masks[i, :count, :count] = 1

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


def permute_graph(graph: pyg.data.Data) -> pyg.data.Data:
    """
    Permutes the nodes of a given PyTorch Geometric graph.
    
    IMPORTANT: this method discards the original graph's edge features
    and don't include them in the permutated graph.

    Args:
        graph (pyg.data.Data): The input graph.

    Returns:
        pyg.data.Data: The permuted graph.
    """
    perm = torch.randperm(graph.num_nodes)
    
    # Permute the node features
    permuted_x = graph.x[perm]
    # Permute the edge indices
    
    permuted_edge_index = permute_edge_index(graph.edge_index, perm)
    # Create the permuted graph
    permuted_graph = pyg.data.Data(x=permuted_x, y=graph.y, edge_index=permuted_edge_index)
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

def evaluate_model(model: torch.nn.Module, dataset: Iterable, batch_size: Optional[int]=32):

    device = torch.device("cuda")

    val_loader = DataLoader(
        dataset, batch_size=batch_size, collate_fn=collate_spectral_dataset
    )
    evaluator = Evaluator(name="ogbg-molhiv")
    y_true = []
    y_pred = []

    with torch.no_grad():
        for batch in tqdm(val_loader):
            features, eigvects, mask, labels = batch
            features = features.to(device)
            eigvects = eigvects.to(device)
            mask = mask.to(device)
            labels = labels.to(device)

            outputs = model(features, eigvects, mask)
            predicted = torch.argmax(outputs, dim=1)

            y_true.append(labels)
            y_pred.append(predicted)

    y_true = torch.concat(y_true).unsqueeze(1)
    y_pred = torch.concat(y_pred).unsqueeze(1)

    roc_auc = evaluator.eval({"y_true": y_true, "y_pred": y_pred})["rocauc"]
    return roc_auc


class Trainer(object):
    """
    A class for training a model.

    Args:
        model (torch.nn.Module): The model to be trained.
        optimizer (torch.optim.Optimizer): The optimizer used for training.
        scheduler (torch.optim.lr_scheduler._LRScheduler): The learning rate scheduler.
        criterion (torch.nn.Module): The loss function.
        device (torch.device): The device to run the training on.
        use_eigenvects (bool, optional): Whether to use eigenvectors in the training process. Defaults to True.
    """

    def __init__(self, model, optimizer, scheduler, criterion, device, experiment_name, use_eigenvects=True, checkpoint_interval=0):
        self.model = model
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.criterion = criterion
        self.device = device
        self.use_eigenvects = use_eigenvects
        self.writer = SummaryWriter(log_dir=f"runs/{experiment_name}")  # Create a SummaryWriter for TensorBoard logging
        self.experiment_name = experiment_name
        self.checkpoint_interval = checkpoint_interval

    def train(self, train_loader, num_epochs):
        """
        Trains the model.

        Args:
            train_loader (torch.utils.data.DataLoader): The data loader for training data.
            num_epochs (int): The number of epochs to train for.
        """
        self.model.train()  # Set the model to training mode
        total_steps = 0

        for epoch in range(num_epochs):
            for features, eigvects, attn_mask, labels in tqdm(train_loader):
                features = features.to(self.device)
                attn_mask = attn_mask.to(self.device)
                labels = labels.flatten().to(self.device)
                
                # Discard eigenvectors encoding if explicitly requested 
                if self.use_eigenvects:
                    eigvects = eigvects.to(self.device)
                else:
                    eigvects = torch.zeros_like(eigvects).to(self.device)
                
                self.optimizer.zero_grad()

                # Forward pass
                outputs = self.model(features, eigvects, attn_mask)
                loss = self.criterion(outputs, labels)

                # Backward and optimize
                loss.backward()
                self.optimizer.step()
                self.scheduler.step()

                total_steps += 1

                # Log the loss to TensorBoard
                self.writer.add_scalar("Loss/train", loss.item(), total_steps)

            if self.checkpoint_interval and (epoch + 1) % self.checkpoint_interval == 0:
                torch.save(self.model, f"{self.experiment_name}_epoch{epoch+1}.pt")
                print("Saved checkpoint")
            print(f"Epoch {epoch+1} completed")
            

        self.writer.close()  # Close the SummaryWriter
