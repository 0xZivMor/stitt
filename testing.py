from ogb.graphproppred import PygGraphPropPredDataset
import torch
import torch.nn as nn
from project.stitt.gat_old import GatGraphClassifier

from torch_geometric.loader import DataLoader

from stitt import Stitt, StittGraphClassifier
from utils import (
    create_spectral_dataset,
    collate_spectral_dataset_no_eigenvects,
    collate_spectral_dataset,
)
from trainer import Trainer

from transformers import get_linear_schedule_with_warmup
import argparse


def main(args):
    dataset = PygGraphPropPredDataset(name="ogbg-molhiv")
    max_graph = max([graph.num_nodes for graph in dataset])

    split_idx = dataset.get_idx_split() 
    
    if torch.cuda.is_available():
        device = torch.device("cuda")

    if torch.backends.mps.is_available():
        device = torch.device("mps")


    batch_size = args.batch_size
    n_heads = args.heads
    n_layers = args.layers
    d_input = args.d_input
    d_attn = args.d_attn
    d_ffn = args.d_ffn
    n_epochs = args.epochs
    r_warmup = args.warmup
    n_classes = args.classes
    lr = args.lr
    model_path = f"{args.name}.pt"
    checkpoint_interval = args.checkpoints
    no_eigenvects = args.no_eigenvects
    model_arch = args.model_arch
    d_atom_emb = args.d_atom_emb

    # print("Training parameters:")
    # for arg in vars(args):
    #     print(f"{arg}: {getattr(args, arg)}")


    if model_arch == "stitt":
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
        if no_eigenvects:
            collate = collate_spectral_dataset_no_eigenvects
        else:
            collate = collate_spectral_dataset

        train_spect_ds = torch.load(args.train_dataset)
        val_ds = torch.load(args.val_dataset)


        train_loader = DataLoader(
            train_spect_ds,
            batch_size=batch_size,
            shuffle=True,
            collate_fn=collate,
            pin_memory=True
        )

        val_loader = DataLoader(
            val_ds, batch_size=batch_size, collate_fn=collate, pin_memory=True,
            num_workers=2
        )



    elif model_arch == "gat" :
        model = GatGraphClassifier(
            d_input=d_input,
            d_ffn=d_ffn,
            n_heads=n_heads,
            n_layers=n_layers,
            n_classes=n_classes,
            d_atom_emb=d_atom_emb,
            device=device,
        )
        # collate = torch.utils.data.default_collate
        train_loader = DataLoader(dataset[split_idx['train']], batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(dataset[split_idx['valid']], batch_size=batch_size, shuffle=False)


    else:
        raise ValueError("Model architecture not supported")

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0)

    training_steps = n_epochs * len(train_loader)
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
        experiment_name=args.name,
        checkpoint_interval=checkpoint_interval
    )

    trainer.train(train_loader, val_loader, n_epochs)

    torch.save(model, model_path)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Script description')
    parser.add_argument('--batch_size', type=int, default=32, help='Batch size')
    parser.add_argument('--heads', type=int, default=8, help='Number of attention heads')
    parser.add_argument('--layers', type=int, default=4, help='Number of transformer layers')
    parser.add_argument('--d_input', type=int, default=256, help='Input dimension')
    parser.add_argument('--d_attn', type=int, default=256, help='Attention dimension')
    parser.add_argument('--d_ffn', type=int, default=128, help='Feed-forward network dimension')
    parser.add_argument('--epochs', type=int, default=5, help='Number of epochs')
    parser.add_argument('--warmup', type=float, default=0.1, help='Warmup ratio')
    parser.add_argument('--classes', type=int, default=2, help='Number of classes')
    parser.add_argument('--lr', type=float, default=5e-5, help='Learning rate')
    parser.add_argument('--name', type=str, default='gat', help='Experiment name')
    parser.add_argument('--checkpoints', type=int, default=0, help='Save model at intervals')
    parser.add_argument('--no_eigenvects', action='store_true',default=False, help='Do not use eigenvectors')
    parser.add_argument('--train_dataset', type=str, help='Path of training dataset file')
    parser.add_argument('--val_dataset', type=str, help='Path of dataset file')
    parser.add_argument('--model_arch',type=str,default='gat', help='model architecture')
    parser.add_argument('--d_atom_emb', type=int, default=100, help='dimension of atom embedding')
    args = parser.parse_args()
    
    main(args)

#----------------------dubug-------------------------------

import yaml

def debug():
    with open('config.yaml', 'r') as f:
        config = yaml.safe_load(f)
        main(config)

debug()
