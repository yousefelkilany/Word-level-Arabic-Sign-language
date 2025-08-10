from torch import nn


class ResidualBiLSTMBlock(nn.Module):
    def __init__(self, hidden_size, drop_prop=0.3):
        super().__init__()

        self.lstm = nn.LSTM(
            hidden_size, hidden_size // 2, batch_first=True, bidirectional=True
        )
        self.layer_norm = nn.LayerNorm(hidden_size)
        self.dropout = nn.Dropout(drop_prop)

    def forward(self, x):
        return self.layer_norm(x + self.dropout(self.lstm(x)[0]))


class AttentionBiLSTM(nn.Module):
    def __init__(
        self,
        input_size,
        hidden_size,
        num_lstm_blocks,
        num_classes,
        dropout_prob=0.3,
        lstm_dropout_prob=0.3,
    ):
        super().__init__()

        self.lstm_proj_layer = nn.Linear(input_size, hidden_size)
        self.lstms = nn.Sequential(
            [
                ResidualBiLSTMBlock(hidden_size, lstm_dropout_prob)
                for _ in range(num_lstm_blocks)
            ]
        )

        self.attention = nn.MultiheadAttention(
            embed_dim=hidden_size,
            num_heads=8,
            dropout=dropout_prob,
            batch_first=True,
        )

        self.attn_layer_norm = nn.LayerNorm(hidden_size)

        self.dropout = nn.Dropout(dropout_prob)
        self.fc = nn.Linear(hidden_size, num_classes)

    def forward(self, x):
        x = self.lstm_proj_layer(x)
        for lstm_block in self.lstms:
            x = lstm_block(x)

        attn_output = self.attn_layer_norm(x + self.attention(x, x, x)[0])

        aggregated_output = attn_output.mean(dim=1)

        dropped_out = self.dropout(aggregated_output)
        logits = self.fc(dropped_out)
        return logits
