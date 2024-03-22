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
import argparse


def main(args):
    dataset = PygGraphPropPredDataset(name="ogbg-molhiv")
    idx = dataset.get_idx_split()
    max_graph = max([graph.num_nodes for graph in dataset])
    device = torch.device("cuda")

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
    upsample = args.upsample
    checkpoint_interval = args.checkpoints

    print("Training parameters:")
    for arg in vars(args):
        print(f"{arg}: {getattr(args, arg)}")
    print(f"max grpah: {max_graph}, device: {device}")
    train_spect_ds = create_spectral_dataset(dataset[idx["train"]], upsample=upsample)
    
    
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
        experiment_name=args.name,
        checkpoint_interval=checkpoint_interval
    )

    trainer.train(train_loader, n_epochs)

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
    parser.add_argument('--name', type=str, default='stitt', help='Experiment name')
    parser.add_argument('--checkpoints', type=int, default=0, help='Save model at intervals')

    args, unknown = parser.parse_known_args()
    parser.add_argument('--upsample', type=int, nargs=args.classes, default=[0] * args.classes)
    args = parser.parse_args(unknown, namespace=args)
    
    main(args)
