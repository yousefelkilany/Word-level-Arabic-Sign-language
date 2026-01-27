# Model Architecture Design

#model #architecture #deep-learning

The core of the recognition system is a custom **Attention-based Bidirectional LSTM (BiLSTM)** network designed for processing streaming skeletal data.

## Architecture Overview

The model takes a sequence of spatial keypoints and outputs a probability distribution over the sign classes.

### 1. Spatial Group Embedding (SGE)
Before temporal processing, we independently project distinct body parts into a shared latent space. This allows the model to learn part-specific features.

- **Inputs**: Pose, Face, Left Hand, Right Hand.
- **Projections**: 4 separate Linear layers.
- **Fusion**: Concatenation -> GELU -> BatchNorm -> Permute.
- **Output**: A unified feature vector per time step.

### 2. Residual BiLSTM Layers
We use a stack of **BiLSTM blocks** to capture temporal dependencies.

- **Bidirectional**: Processes the sequence forwards and backwards to capture context.
- **Residual Connection**: The input to the block is added to the output to prevent vanishing gradients.
- **Layer Normalization**: Applied after the residual addition for stability.

### 3. Self-Attention Pooling
Instead of simply taking the last hidden state (which loses early context) or averaging all states (which dilutes information), we use a **Self-Attention** mechanism.

- **Query**: The model learns to weigh each time step based on its relevance.
- **Weighted Sum**: The final representation is a weighted sum of all time steps.

### 4. Classification Head
- **Dropout**: For regularization.
- **Linear Layer**: Maps the pooled representation to `num_classes` logits.

## Diagram

```mermaid
graph TD
    Input[Input Sequence (T, F)] --> SGE[Spatial Group Embedding]
    SGE --> L1[ResBiLSTM Block 1]
    L1 --> L2[ResBiLSTM Block 2]
    L2 --> L3[ResBiLSTM Block 3]
    L3 --> L4[ResBiLSTM Block 4]
    L4 --> Attn[Multihead Attention]
    Attn --> Pool[Attention Pooling]
    Pool --> FC[Classifier Head]
    FC --> Softmax[Softmax Probabilities]
```

## Related Documentation

- [[source/modelling/model_py|model.py Source Code]]
- [[models/training_process|Training Process]]
