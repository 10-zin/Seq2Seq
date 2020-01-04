import torch
import torch.nn as nn
import random
class Encoder(nn.Module):
    def __init__(self, input_dim, emb_dim, hid_dim, n_layers, dropout):
        super().__init__()
        self.hid_dim = hid_dim
        self.n_layers = n_layers
#         print(type(emb_dim), type(hid_dim), type(n_layers), type(dropout))
        self.embedding = nn.Embedding(input_dim, emb_dim)
        self.rnn = nn.LSTM(emb_dim, hid_dim, n_layers, dropout = dropout)
        self.dropout = nn.Dropout(dropout)
    def forward(self, src):
        
        embedded = self.dropout(self.embedding(src))
        outputs, (hidden, cell) = self.rnn(embedded)
        return hidden, cell      
        

class Decoder(nn.Module):
    def __init__(self, output_dim, emb_dim, hid_dim, n_layers, dropout):
        super().__init__()
        self.hid_dim = hid_dim
        self.n_layers = n_layers
        self.output_dim = output_dim
        self.embedding = nn.Embedding(output_dim, emb_dim)
        self.rnn = nn.LSTM(emb_dim, hid_dim, n_layers, dropout= dropout)
        self.dropout = nn.Dropout(dropout)
        self.out = nn.Linear(hid_dim, output_dim)
    def forward(self, input, hidden, cell):
        input = input.unsqueeze(0)
        embedded = self.dropout(self.embedding(input))
        output, (hidden, cell) = self.rnn(embedded, (hidden, cell))
        prediction = self.out(output.squeeze(0))
        
        return prediction, hidden, cell
        

class Seq2Seq(nn.Module):
    def __init__(self, encoder, decoder, device):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.device = device
        
        assert encoder.hid_dim == decoder.hid_dim
        assert encoder.n_layers == decoder.n_layers
        
    def forward(self, src, trg, teacher_forcing_ratio = 0.5):
            batch_size = trg.shape[1]
            max_len = trg.shape[0]
            trg_vocab_size = self.decoder.output_dim
            
            outputs = torch.zeros(max_len, batch_size, trg_vocab_size).to(self.device)
            
            hidden, cell = self.encoder(src)
            
            input = trg[0,:]
            
            for t in range(1, max_len):
                output, hidden, cell = self.decoder(input, hidden, cell)
                outputs[t] = output
                teacher_force = random.random() < teacher_forcing_ratio
                top1 = output.argmax(1)
                input = trg[t] if teacher_force else top1
            
            return outputs            
             

def init_weights(m):
    for name, param in m.named_parameters():
        nn.init.uniform_(param.data, -0.08, 0.08)


def enc_dec(inp_dim, out_dim, emb_dim, hid_dim, drop_prob, n_layers, device):
    enc = Encoder(inp_dim, emb_dim, hid_dim, n_layers, drop_prob)
    dec = Decoder(out_dim, emb_dim, hid_dim, n_layers, drop_prob)

    model = Seq2Seq(enc, dec, device).to(device)
    model.apply(init_weights)

    return model