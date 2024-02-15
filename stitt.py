import torch
import torch_geometric
import scipy.sparse as sp

def compute_normalized_laplacian_eig(graph):
  """
  Compute the eigenvalues and eigenvectors of the normalized Laplacian matrix of a graph.

  Args:
    graph (torch_geometric.data.Data): The input graph.

  Returns:
    tuple: A tuple containing the sorted eigenvalues and eigenvectors of the normalized Laplacian matrix.
  """
  
  laplacian = torch_geometric.utils.get_laplacian(graph.edge_index, normalization="sym")
  row, col = laplacian[0]
  sparse_lap = sp.coo_matrix((laplacian[1], (row, col)), (graph.num_nodes, graph.num_nodes))
  
  # Eigendecomposition on sparse matrices is not complete
  lap = torch.tensor(sparse_lap.toarray())

  # Compute the eigendecomposition of the Laplacian
  eigenvalues, eigenvectors = torch.linalg.eigh(lap)
  eigenvalues = torch.real(eigenvalues)
  eigenvectors = torch.real(eigenvectors)
  
  # S
  sorted_indices = torch.argsort(eigenvalues, descending=True)
  eigenvalues = eigenvalues[sorted_indices]
  eigenvectors = eigenvectors[:, sorted_indices]

  return eigenvalues, eigenvectors
