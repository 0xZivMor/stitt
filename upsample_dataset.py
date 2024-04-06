from ogb.graphproppred import PygGraphPropPredDataset

import argparse
import utils
import torch

def main():
    # Create an ArgumentParser object
    parser = argparse.ArgumentParser(description='Upsample dataset based on ogbg-molhiv')

    # Add command line arguments
    parser.add_argument('-o', '--output', type=str, help='Path to save the upsampled dataset')
    parser.add_argument('--upsample', type=int, nargs=2, help='Number of times to upsample each class')

    # Parse the command line arguments
    args = parser.parse_args()

    dataset = PygGraphPropPredDataset("ogbg-molhiv")
    idx = dataset.get_idx_split()
    print("Loaded dataset")
    
    # Call utils.create_spectral_dataset with the provided arguments
    ds = utils.create_spectral_dataset(dataset[idx['train']],
                                       args.upsample)
    torch.save(ds, args.output)

if __name__ == '__main__':
    main()