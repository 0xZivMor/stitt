from ogb.graphproppred import PygGraphPropPredDataset, Evaluator
import torch
import torch.nn as nn


from torch.utils.data import DataLoader

from stitt import Stitt, StittGraphClassifier
from utils import (
    create_spectral_dataset,
    Trainer,
    collate_spectral_dataset_no_eigenvects,
    collate_spectral_dataset
)

from transformers import get_linear_schedule_with_warmup
from tqdm import tqdm

dataset = PygGraphPropPredDataset(name="ogbg-molhiv")
idx = dataset.get_idx_split()
max_graph = max([graph.num_nodes for graph in dataset])
device = torch.device("cuda")

batch_size = 32
n_heads = 8
n_layers = 4
d_input = 256
d_attn = 256
d_ffn = 128
n_epochs = 5
r_warmup = 0.1
n_classes = 2
lr = 5e-5


train_spect_ds = create_spectral_dataset(dataset[idx["train"]], upsample=[15, 40])
model = StittGraphClassifier(
    d_input=d_input,
    d_attn=d_attn,
    d_ffn=d_ffn,
    max_graph=max_graph,
    n_heads=n_heads,
    n_layers=n_layers,
    n_classes=n_classes,
    device=device,
)


train_loader = DataLoader(
    train_spect_ds,
    batch_size=batch_size,
    shuffle=True,
    collate_fn=collate_spectral_dataset,
)
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0)

training_steps = n_epochs * len(train_spect_ds)
scheduler = get_linear_schedule_with_warmup(
    optimizer,
    num_warmup_steps=r_warmup * training_steps,
    num_training_steps=training_steps,
)

trainer = Trainer(
    model.to(device),
    optimizer=optimizer,
    scheduler=scheduler,
    criterion=criterion,
    device=device,
)

trainer.train(train_loader, n_epochs)

torch.save(model, "stitt_molhiv_3.pt")
