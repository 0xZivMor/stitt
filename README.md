# Stitt
This is the source code for our final project in Technion 236004 - Topics in Transformers and Attention.

To train Stitt, use
```
python testing.py --layers [NUMBER OF LAYERS] --d_input [INPUT_DIM] --d_attn [ATTENTION_DIM] --d_ffn [FEED FORWARD DIM] --epochs [EPOCHS] --name [EXPERIMENT NAME] --train_dataset [PATH OF CUSTOM TRAINING DATASET] --val_dataset [PATH OF CUSTOM VALIDATION DATASET]
```
Set the arguments as you see fit. TensorBoard report is created automatically under `runs\[EXPERIMENT NAME]`.

In order to create the custom dataset, use `python update_dataset.py` and follow the instructions.
