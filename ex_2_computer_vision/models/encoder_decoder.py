import torch.nn as nn


class EncoderDecoder(nn.Module):
    def __init__(self, encoder, decoder):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder
    
    def forward(self, x):
        B, _, H, W = x.shape
        encoder_features = self.encoder(x)
        logits = self.decoder(encoder_features)
        return nn.functional.interpolate(logits, size=(H, W), mode='bilinear')

    def train(self, train=True):
        self.decoder.train(train)
    
