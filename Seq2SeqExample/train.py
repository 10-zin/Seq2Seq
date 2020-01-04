import data_iterator, model
import torch
import torch.nn as nn
import random
import argparse
import torch.optim as optim
from tqdm import tqdm

def training(model, iterator, optimizer, criterion, clip):
    model.train()
    epoch_loss = 0
    
    for batch in tqdm(iterator):
        src = batch.src
        trg = batch.trg
        
        optimizer.zero_grad()
        
        output = model(src, trg)
        output = output[1:].view(-1, output.shape[-1])
        trg = trg[1:].view(-1)
        
        loss = criterion(output, trg)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), clip)
        optimizer.step()
        epoch_loss+=loss.item()
     
    
    return epoch_loss/len(iterator)

def evaluate(model, iterator, criterion):
    
    model.eval()
    
    epoch_loss = 0
    
    with torch.no_grad():
    
        for batch in iterator:

            src = batch.src
            trg = batch.trg

            output = model(src, trg, 0) #turn off teacher forcing

            #trg = [trg sent len, batch size]
            #output = [trg sent len, batch size, output dim]

            output = output[1:].view(-1, output.shape[-1])
            trg = trg[1:].view(-1)

            #trg = [(trg sent len - 1) * batch size]
            #output = [(trg sent len - 1) * batch size, output dim]

            loss = criterion(output, trg)
            
            epoch_loss += loss.item()
             
    return epoch_loss / len(iterator)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--hid_dim', help = "Enter the decoder's hidden dimension", type = int, default = 512)
    parser.add_argument('--emb_dim', help = "Enter the embedding dimension", type = int, default = 300)
    parser.add_argument('--n_layers', help = "Enter the number of layers", type = int, default = 2)
    parser.add_argument('--drop_prob', help = "Enter the dropout probability", type = float, default = 0.5)
    parser.add_argument('--batch_size', help = "Enter the batch size", type = int, default = 64)
    parser.add_argument('--epochs', help = "Enter the number of epochs", type = int, default = 4)
    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    train_iterator, valid_iterator, test_iterator, src, trg = data_iterator.get_data_iterator(args.batch_size, device)
    criterion = nn.CrossEntropyLoss(ignore_index = trg.vocab.stoi['<pad>'])
    CLIP = 1
    best_valid_loss = float('inf')
    # print('ok')

    m = model.enc_dec(len(src.vocab), len(trg.vocab), args.emb_dim, args.hid_dim, args.drop_prob, args.n_layers, device)
    print(m)

    optimizer = optim.Adam(m.parameters())

    for epoch in range(args.epochs):
        train_loss = training(m, train_iterator, optimizer, criterion, CLIP)
        valid_loss = evaluate(m, valid_iterator, criterion)
        if valid_loss < best_valid_loss:
            best_valid_loss = valid_loss
            torch.save(m.state_dict(), 'seqmodel.pth')
        print(f'epoch : {epoch}')
        print(f'Train_Loss: {train_loss:.3f}')
        print(f'Valid_Loss: {valid_loss:.3f}')
    test_loss = evaluate(m, test_iterator, criterion)
    print(f'| Test Loss: {test_loss:.3f}')