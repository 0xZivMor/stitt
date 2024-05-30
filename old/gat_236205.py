import torch
from torch_geometric.loader import NeighborLoader
from torch_geometric.datasets import Planetoid

dataset    = Planetoid(root='.', name="Pubmed")
graph_data = dataset[0]
print(f'Num_graphs: {len(dataset)}')
print(f'Number_vertex: {graph_data.x.shape[0]}')
print(f'Number_features: {dataset.num_features}')
print(f'Number_classes: {dataset.num_classes}')
import torch.nn.functional as F
from torch.nn import Linear, Dropout
from torch_geometric.nn import GATConv, GATv2Conv

train_loader = NeighborLoader(
    graph_data,
    num_neighbors = [10, 20],
    batch_size    = 32,
    input_nodes   = graph_data.train_mask,
)

def acc(test_label, true_label):
    return ((test_label == true_label).sum() / len(true_label)).item()
  

def test_acc(train_model, data):
    train_model.eval()
    _, out__ = train_model(data.x, data.edge_index)
    acc_value      = acc(out__.argmax(dim=1)[data.test_mask], data.y[data.test_mask])

    return acc_value

class GraphAttentionNet(torch.nn.Module):
  def __init__(self, dim_input, dim_hidden, dim_output, heads):
    super().__init__()
    self.layer_1 = GATConv(dim_input,          dim_hidden, heads = heads)
    self.layer_2 = GATConv(dim_hidden * heads, dim_output, heads = heads)
    self.optimizer = torch.optim.Adam(self.parameters(),
                                      lr=1e-3,
                                      weight_decay=1e-5)
    
  def forward(self, point, edge_info):

    h_v = F.dropout(point, p = 0.25, training=self.training)
    h_v = self.layer_1(h_v, edge_info)
    h_v = torch.relu(h_v)
    h_v = F.dropout(h_v, p = 0.25, training=self.training)
    h_v = self.layer_2(h_v, edge_info)
    
    return h_v, F.log_softmax(h_v, dim=1)

  def fit(self,  graph_data, num_epoch):

    loss_eva  = torch.nn.CrossEntropyLoss()
    optimizer = self.optimizer


    self.train()
    for epoch in range(num_epoch + 1):

        optimizer.zero_grad()
        _, out__ = self(graph_data.x, graph_data.edge_index)
        loss     = loss_eva(out__[graph_data.train_mask], graph_data.y[graph_data.train_mask])
        accuracy = acc(out__[graph_data.train_mask].argmax(dim=1), graph_data.y[graph_data.train_mask])

        loss.backward()
        optimizer.step()


        validation_loss      = loss_eva(out__[graph_data.val_mask], graph_data.y[graph_data.val_mask])
        validatioin_accuracy = acc(out__[graph_data.val_mask].argmax(dim=1), graph_data.y[graph_data.val_mask])

        if(epoch % 5 == 0):
          print(f'Epoch {epoch}  training_loss: {loss:.2f} '
                f' training_accuracy: {accuracy*100:.2f}%  validation_loss: '
                f'{validation_loss:.2f}  validatioin_accuracy: '
                f'{validatioin_accuracy*100:.2f}%')
          
gan_model = GraphAttentionNet(dataset.num_features, 64, dataset.num_classes, 8)
print(gan_model)

gan_model.fit(graph_data, 150)
print(f'Test accuracy: {test_acc(gan_model, graph_data)*100:.2f} %\n')