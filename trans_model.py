import torch
import torch.nn as nn

class Transformer(nn.Module):
    def __init__(
        self,
        embedding_size,
        src_embed_size,
        trg_embed_size,
        num_heads,
        num_encoder_layers,
        num_decoder_layers,
        forward_expansion,
        dropout,
        device,
    ):
        super(Transformer, self).__init__()
        
        self.embedding_size = embedding_size
        self.src_embed_size = src_embed_size
        self.trg_embed_size = trg_embed_size
        
        self.device = device
        self.transformer = nn.Transformer(
            embedding_size,
            num_heads,
            num_encoder_layers,
            num_decoder_layers,
            forward_expansion,
            dropout
        )
        
        self.fc_out = nn.Linear(embedding_size, trg_embed_size)
        self.dropout = nn.Dropout(dropout)

    def forward(self, src, trg):
        embed_src = self.dropout(src)
        embed_trg = self.dropout(trg)

        out = self.transformer(
            embed_src,
            embed_trg
        )
        
        out = self.fc_out(out)
        return out
    
# def test():
#     x = torch.rand((10, 1, 512))
#     y = torch.rand((10, 1, 512))
#     model = Transformer(512,512,512,8,3,3,4,0.3,'cpu')
#     preds = model(x,y)
#     print(preds.shape)
    
# if __name__ == "__main__":
#     test()
    