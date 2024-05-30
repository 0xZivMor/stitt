from ogb.graphproppred import PygGraphPropPredDataset
import torch
import torch.nn as nn
from gat import GatGraphClassifier
from torch_geometric.loader import DataLoader
from stitt import StittGraphClassifier
from trainer import Trainer
from transformers import get_linear_schedule_with_warmup
from torchsampler import ImbalancedDatasetSampler
import argparse
from utils import (
    collate_spectral_dataset_no_eigenvects,
    collate_spectral_dataset,
)

def main(args):

    #saving the hyerparameters
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

    #downloading the dataset 
    dataset = PygGraphPropPredDataset(name="ogbg-molhiv")
    split_idx = dataset.get_idx_split() 

    #setting the device 
    device = torch.device('cpu')
    if torch.cuda.is_available():
        device = torch.device("cuda")


    #creating the chosen model
    if model_arch == "stitt":
        max_graph = max([graph.num_nodes for graph in dataset])
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
        #we use the new dataset we created which contanins the eigenvectors and after balancing the dataset 
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
            d_hidden=d_input,
            n_heads=n_heads,
            n_layers=n_layers,
            n_classes=n_classes,
            device=device
        )

        #we use the default dataset and balance it 
        balanced_sampler = ImbalancedDatasetSampler(dataset[split_idx['train']], labels= [data.y.item() for data in dataset[split_idx['train']]])
        train_loader = DataLoader(dataset[split_idx['train']],batch_size=batch_size, sampler=balanced_sampler)
        val_loader = DataLoader(dataset[split_idx['valid']], batch_size=batch_size, shuffle=False)

    else:
        raise ValueError("Model architecture not supported")

    #setting the training process
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)

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
    parser.add_argument('--heads', type=int, default= 8, help='Number of attention heads')
    parser.add_argument('--layers', type=int, default=4, help='Number of transformer layers')
    parser.add_argument('--d_input', type=int, default=512, help='Input dimension')
    parser.add_argument('--d_attn', type=int, default=256, help='Attention dimension')
    parser.add_argument('--d_ffn', type=int, default=128, help='Feed-forward network dimension')
    parser.add_argument('--epochs', type=int, default=60, help='Number of epochs')
    parser.add_argument('--warmup', type=float, default=0.1, help='Warmup ratio')
    parser.add_argument('--classes', type=int, default=2, help='Number of classes')
    parser.add_argument('--lr', type=float, default=5e-4, help='Learning rate')
    parser.add_argument('--name', type=str, default='gat', help='Experiment name')
    parser.add_argument('--checkpoints', type=int, default=0, help='Save model at intervals')
    parser.add_argument('--no_eigenvects', action='store_true',default=False, help='Do not use eigenvectors')
    parser.add_argument('--train_dataset', type=str,default='./dataset/ogbg_molhiv/molhiv_spectral_dataset_train.pt', help='Path of training dataset file')
    parser.add_argument('--val_dataset', type=str,default='./dataset//ogbg_molhiv/molhiv_spectral_dataset_val.pt', help='Path of dataset file')
    parser.add_argument('--model_arch',type=str,default='gat', help='model architecture')
    args = parser.parse_args()
    
    main(args)

