import torch
import torch.nn as nn
import numpy as np
import random
class Encoder(nn.Module):
    
    def __init__(self, inp_vocab_dim, enc_hid_dim, embed_dim, dec_hid_dim, drop_prob):
        super().__init__()

        self.embed = nn.Embedding(inp_vocab_dim, embed_dim)
        self.gru = nn.GRU(embed_dim, enc_hid_dim, bidirectional = True)
        
        # concatenate two bidirectional hidden vectors and pass through a linear layer to generate one hidden 
        # vector of decoder hidden size
        self.linear = nn.Linear(enc_hid_dim*2, dec_hid_dim) 
        self.dropout = nn.Dropout(drop_prob)
        
    def forward(self, inp_sntc):

        inp_embed = self.dropout(self.embed(inp_sntc))
        outputs, hidden = self.gru(inp_embed)

        open_hidden = torch.cat((hidden[0, :, :], hidden[1, :, :]), dim = 1)
        hidden = torch.tanh(self.linear(open_hidden))
        
        return outputs, hidden   
    
class Attention(nn.Module):
    
    def __init__(self, enc_hid_dim, dec_hid_dim):
        super().__init__()
        
        self.sim = nn.Linear((enc_hid_dim*2) + dec_hid_dim, dec_hid_dim)
        self.weight = nn.Parameter(torch.rand(dec_hid_dim))
        
    def forward(self, curr_dec_hid, enc_outputs):
        
        sntc_length = enc_outputs.shape[0]
        batch_size = enc_outputs.shape[1]
        
        curr_dec_hid = curr_dec_hid.unsqueeze(1).repeat(1, sntc_length, 1) 
        enc_outputs =  enc_outputs.permute(1, 0, 2)
        
        e = torch.tanh(self.sim(torch.cat((curr_dec_hid, enc_outputs), dim = 2))) 
        e = e.permute(0, 2, 1)

        weight = self.weight.repeat(batch_size, 1).unsqueeze(1)

        attn_dist = torch.bmm(weight, e).squeeze(1)
        norm_attn = torch.softmax(attn_dist, dim = 1)
        
        return norm_attn
    
class Decoder(nn.Module):
    
    def __init__(self, vocab_dim, embed_dim, attn, enc_hid_dim, dec_hid_dim, drop_prob ):
        super().__init__()
        
        self.embed = nn.Embedding(vocab_dim, embed_dim)
        self.attn = attn
        
        self.vocab_dim = vocab_dim
        self.gru = nn.GRU((enc_hid_dim*2) + embed_dim, dec_hid_dim)
        
        self.linear = nn.Linear(enc_hid_dim*2 + dec_hid_dim + embed_dim, vocab_dim)
        self.dropout = nn.Dropout(drop_prob)
    
    def forward(self, inp, dec_hidden, enc_outputs):
        
        inp = inp.unsqueeze(0)
        embedding = self.dropout(self.embed(inp))
        
        norm_attn = self.attn(dec_hidden, enc_outputs)    
        norm_attn = norm_attn.unsqueeze(1)
        
        enc_outputs = enc_outputs.permute(1, 0, 2)

        weighted_sum = torch.bmm(norm_attn, enc_outputs)
        weighted_sum = weighted_sum.permute(1, 0, 2)

        output, hidden = self.gru(torch.cat((embedding, weighted_sum), dim = 2), dec_hidden.unsqueeze(0) )
        
        assert (output == hidden).all()
        
        embedding = embedding.squeeze(0)
        output = output.squeeze(0)
        weighted_sum = weighted_sum.squeeze(0)
        
        next_word = self.linear(torch.cat((output, weighted_sum, embedding), dim = 1))
        
        return next_word, hidden.squeeze(0)

class seq2seq(nn.Module):
    
    def __init__(self, device, encoder, decoder):
        super().__init__()
        self.device = device
        self.encoder = encoder
        self.decoder = decoder
    def forward(self, sntc_input, sntc_label, thresh = 0.5):
        
        sntc_input = sntc_input.permute(1, 0)
        sntc_label = sntc_label.permute(1, 0)
        
        enc_outputs, hidden = self.encoder(sntc_input)
        
        label_len = sntc_label.shape[0]
        batch_size = sntc_input.shape[1]

        vocab_dim = self.decoder.vocab_dim
        
        dec_outputs = torch.zeros(label_len, batch_size, vocab_dim).to(self.device)
        input_word = sntc_label[0, :]
        
        for i in range(1, label_len):
            output, hidden = self.decoder(input_word, hidden, enc_outputs)
            dec_outputs[i] = output
            pred_next_word = output.argmax(1)
            input_word = sntc_label[i] if random.random() < thresh else pred_next_word
        
        return dec_outputs   


def init_weights(m):
    for name, param in m.named_parameters():
        if 'weight' in name:
            nn.init.normal_(param.data, mean = 0, std = 0.01)
        else:
            nn.init.constant_(param.data, 0)

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def enc_dec_attn(enc_hid_dim, dec_hid_dim, embed_dim, drop_prob, device, inp_vocab_dim, label_vocab_dim):
    enc = Encoder(inp_vocab_dim, enc_hid_dim, embed_dim, dec_hid_dim, drop_prob)
    attn = Attention(enc_hid_dim, dec_hid_dim)
    dec = Decoder(label_vocab_dim, embed_dim, attn, enc_hid_dim, dec_hid_dim, drop_prob)

    model = seq2seq(device, enc, dec).to(device)
    model.apply(init_weights)
    return model
