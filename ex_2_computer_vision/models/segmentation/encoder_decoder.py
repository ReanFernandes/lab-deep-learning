import torch.nn as nn

# Encoder-Decoder wrapper for segmentation.
# - executes encoder and decoder modules in sequence
# - upsamples the decoder output to the size of the original input, via bilinear interpolation
# - only sets the decoder parameters as trainable, keeping the encoder frozen

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
    
