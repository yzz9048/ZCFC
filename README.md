# ZCFC

## Entry Point and Overall Workflow

```
main.py: Set the random seed, parse arguments, and select the device
  └─ BaseRCATrainer: Load SN(A) / TT(B) data and initialize the model
       ├─ --train: Labeled source-domain samples + normal samples from both domains → Train → Save weights and prototypes
       └─ Default: Load weights → Recompute prototypes using training and normal data → Test on A and B → Report metrics
```

Key files:

| File | Purpose |
| --- | --- |
| `main.py` | Command-line arguments, random seed, and training/testing branches |
| `model/RCADataLoader.py` | Load pickle files, normalize data, and create DataLoaders |
| `model/baseRCA.py` | Model wrapper, loss functions, training, prototype computation, and evaluation |
| `model/encoder.py` | Multimodal encoding, entity feature fusion, and graph construction |
| `model/gnn.py` | Graph neural network; the current forward pass uses GraphSAGE |
| `model/decoder.py` | Dual output heads for fault detection and fault-type classification |
| `shared_util/` | Utilities for random seeds, logging, files, and other tasks |

## Training

Run the following example in PowerShell from the project root directory. First, replace the dataset paths and create the output directory. During training, `--model_path` is a **filename prefix**: every 50 epochs, the code appends the epoch number and `.pt` directly to this prefix. `--prototype_path` must point to a file, not a directory.

```powershell
New-Item -ItemType Directory -Force -Path .\checkpoints | Out-Null
python main.py --train `
  --dataset_path_SN "../sn.pkl" `
  --dataset_path_TT "../tt.pkl" `
  --model_path ".\checkpoints\model_" `
  --prototype_path ".\checkpoints\prototypes.pt" `
  --epochs 200 --batch_size 64 --gpu 0
```

## Testing

Omit `--train` to enter testing mode. In this case, `--model_path` must be the **full path to the model weights file**, rather than the prefix used during training:

```powershell
python main.py `
  --dataset_path_SN "../sn.pkl" `
  --dataset_path_TT "../tt.pkl" `
  --model_path ".\checkpoints\model_200.pt" `
  --batch_size 64 --gpu 0
```


## Main Parameters

| Parameter | Default | Description |
| --- | --- | --- |
| `--train` | Not set | Train when specified; otherwise, run testing |
| `--dataset_path_SN` / `--dataset_path_TT` | See `main.py` | Source domain A and target domain B, respectively |
| `--model_path` | `./model/snTott/200.pt` | Filename prefix for saving during training; full filename during testing |
| `--prototype_path` | `./model/snTott/proto/` | Path to the prototype file during training; the default is a directory and should be overridden with a filename; not read during testing |
| `--window_size` / `--llm_embedding` | `10` / `1024` | Time-window length and log embedding dimension; must match the data |
| `--gpu` | `0` | Use the specified GPU index when CUDA is available; otherwise, use the CPU |
| `--epochs` / `--batch_size` | `200` / `64` | Number of training epochs and the batch size shared across data splits |
| `--lr` / `--weight_decay` | `0.0001` / `0.001` | Adam learning rate and weight decay |
| `--orl_te_heads` / `--orl_te_layers` / `--orl_te_in_channels` | `4` / `2` / `128` | Transformer settings for the metric and trace modalities |
| `--efi_in_dim` / `--efi_te_heads` / `--efi_te_layers` / `--efi_out_dim` | `128` / `2` / `1` / `256` | Entity feature encoding and alignment settings |
| `--eff_in_dim` / `--eff_out_dim` | `256` / `128` | Graph network input and output dimensions |
