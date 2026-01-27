# parallel_train.py

#source #modelling #distributed

Launcher script for distributed training.

## Functionality
- Uses `torch.distributed.run` (or similar mechanism) to spawn multiple processes.
- Invokes `train.py` with the appropriate rank and world size configuration.

## Usage
```bash
make parallel_train
```

## Related Code

- [[source/modelling/train_py|train.py]]
