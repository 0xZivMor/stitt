#!/bin/bash

# Parse command-line arguments
parser() {
    local description='Script description'
    local batch_size=32
    local heads=8
    local layers=4
    local d_input=256
    local d_attn=256
    local d_ffn=128
    local epochs=5
    local warmup=0.1
    local classes=2
    local lr=0.00005
    local name='stitt'
    local checkpoints=0
    local train_dataset=''
    local val_dataset=''
    local model_arch='gat'

    # Call the Python script with the parsed arguments
    python3 -m pdb testing.py \
        --batch_size "$batch_size" \
        --heads "$heads" \
        --layers "$layers" \
        --d_input "$d_input" \
        --d_attn "$d_attn" \
        --d_ffn "$d_ffn" \
        --epochs "$epochs" \
        --warmup "$warmup" \
        --classes "$classes" \
        --lr "$lr" \
        --name "$name" \
        --checkpoints "$checkpoints" \
        --no_eigenvects \
        --train_dataset "$train_dataset" \
        --val_dataset "$val_dataset" \
        --model_arch "$model_arch"
}

# Call the parser function with the provided arguments
parser "$@"
