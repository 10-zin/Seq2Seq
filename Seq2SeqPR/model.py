import torch
import torch.nn as nn
import random
class Encoder(nn.Module):
    def __init__(self, input_dim, emb_dim, hid_dim, dropout):
        super().__init__()
        self.hid_dim = hid_dim
        self.dropout = nn.Dropout(dropout)
        self.embedding = nn.Embedding(input_dim, emb_dim)
        self.rnn = nn.GRU(emb_dim, hid_dim)
    def forward(self, src):
        
        embedded = self.dropout(self.embedding(src))
        outputs, hidden = self.rnn(embedded)
        return hidden      
        

class Decoder(nn.Module):
    def __init__(self, output_dim, emb_dim, hid_dim, dropout):
        super().__init__()
        self.hid_dim = hid_dim
        self.dropout = nn.Dropout(dropout)
        self.output_dim = output_dim
        self.embedding = nn.Embedding(output_dim, emb_dim)
        self.rnn = nn.GRU(emb_dim + hid_dim, hid_dim)
        self.out = nn.Linear(hid_dim*2+emb_dim, output_dim)
    def forward(self, input, hidden, context):
        input = input.unsqueeze(0)
        embedded = self.dropout(self.embedding(input))
        emb_con = torch.cat((embedded, context), dim = 2)
        output, hidden = self.rnn(emb_con, hidden)
        output = torch.cat((embedded.squeeze(0), context.squeeze(0), hidden.squeeze(0)), dim = 1)
        prediction = self.out(output)
        
        return prediction, hidden

class Seq2Seq(nn.Module):
    def __init__(self, encoder, decoder, device):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.device = device
        
        assert encoder.hid_dim == decoder.hid_dim
        
    def forward(self, src, trg, teacher_forcing_ratio = 0.5):
            batch_size = trg.shape[1]
            max_len = trg.shape[0]
            trg_vocab_size = self.decoder.output_dim
            
            outputs = torch.zeros(max_len, batch_size, trg_vocab_size).to(self.device)
            
            context = self.encoder(src)
            hidden = context
            input = trg[0,:]
            
            for t in range(1, max_len):
                output, hidden = self.decoder(input, hidden, context)
                outputs[t] = output
                teacher_force = random.random() < teacher_forcing_ratio
                top1 = output.argmax(1)
                input = trg[t] if teacher_force else top1
            
            return outputs  

def init_weights(m):
    for name, param in m.named_parameters():
        nn.init.uniform_(param.data, -0.08, 0.08)


def enc_dec_pr(inp_dim, out_dim, emb_dim, hid_dim, drop_prob, device):
    enc = Encoder(inp_dim, emb_dim, hid_dim, drop_prob)
    dec = Decoder(out_dim, emb_dim, hid_dim, drop_prob)

    model = Seq2Seq(enc, dec, device).to(device)
    model.apply(init_weights)

    return model