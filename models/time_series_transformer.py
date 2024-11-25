import torch.nn as nn

class TimeSeriesTransformer(nn.Module):
    def __init__(self, input_size, num_layers, num_heads, d_model, dim_feedforward):
        super(TimeSeriesTransformer, self).__init__()
        self.encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=num_heads, dim_feedforward=dim_feedforward
        )
        self.transformer_encoder = nn.TransformerEncoder(self.encoder_layer, num_layers=num_layers)
        self.fc = nn.Linear(d_model, 1)

    def forward(self, x):
        x = self.transformer_encoder(x)
        x = self.fc(x)
        return x
