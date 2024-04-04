from ogb.graphproppred import PygGraphPropPredDataset
from utils import create_spectral_dataset, evaluate_model

import argparse
import torch

def main(args):

    batch_size = args.batch_size
    model_path = args.model_path
    dataset_path = args.dataset
    no_eigenvects = args.no_eigenvects

    if not dataset_path:
        dataset = PygGraphPropPredDataset(name='ogbg-molhiv')       
        idx = dataset.get_idx_split()
        val_spect_ds = create_spectral_dataset(dataset[idx["valid"]])
    else:
        val_spect_ds = torch.load(dataset_path)

    model = torch.load(model_path)
    roc_auc = evaluate_model(model, val_spect_ds, batch_size, no_eigenvects)

    print(f"ROC-AUC: {roc_auc:.3f}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--model_path", type=str, help="Path to the model")
    parser.add_argument("--dataset", type=str, help="Path of spectral dataset to evaluate the model against", default="")
    parser.add_argument('--no_eigenvects', action='store_true', help='Do not use eigenvectors')
    args = parser.parse_args()

    main(args)
