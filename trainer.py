import torch

from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter

from utils import evaluate_model


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

    def __init__(
        self,
        model,
        optimizer,
        scheduler,
        criterion,
        device,
        experiment_name,
        use_eigenvects=True,
        checkpoint_interval=0,
    ):
        self.model = model
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.criterion = criterion
        self.device = device
        self.use_eigenvects = use_eigenvects
        self.writer = SummaryWriter(
            log_dir=f"runs/{experiment_name}"
        )  # Create a SummaryWriter for TensorBoard logging
        self.experiment_name = experiment_name
        self.checkpoint_interval = checkpoint_interval

    def train(self, train_loader, val_loader, num_epochs):
        """
        Trains the model.

        Args:
            train_loader (torch.utils.data.DataLoader): The data loader for training data.
            num_epochs (int): The number of epochs to train for.
        """
        self.model.train()  # Set the model to training mode
        total_steps = 0

        for epoch in range(num_epochs):
            for current_batch in tqdm(train_loader):
                # our model does not use edge_features so we don't use it at all.
                edges_connectivity, edge_features, nodes_features, graph_labels, _ , _ , _ = current_batch
                nodes_features = nodes_features[1].to(self.device) # extracting only the tenzor from the object
                edges_connectivity = edges_connectivity[1].to(self.device) # extracting only the tenzor from the object
                graph_labels = graph_labels[1].flatten().to(self.device)

                # # Discard eigenvectors encoding if explicitly requested
                # if self.use_eigenvects:
                #     eigvects = eigvects.to(self.device)
                # else:
                #     eigvects = torch.zeros_like(eigvects).to(self.device)

                self.optimizer.zero_grad()

                # Forward pass
                outputs = self.model(nodes_features, edges_connectivity)
                loss = self.criterion(outputs, graph_labels)

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

            auroc = evaluate_model(self.model, val_loader, 32, not self.use_eigenvects)

            self.writer.add_scalar("AUROC/validation", auroc, total_steps)

            print(f"Epoch {epoch+1} completed")

        self.writer.close()  # Close the SummaryWriter
