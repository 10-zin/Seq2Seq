import torch
import torch.nn as nn
from tqdm import tqdm
import torch.optim as optim

def val_test(model, data_loader, criterion, w2i_eng_train, i2w_eng,
     w2i_fre_train, i2w_fre, device, status):
        
        total_loss = 0
        model.eval()
        
        for inp, label in data_loader:
#             for token equivalence btw train and val or test
            if status == 'val' :
                i2w_eng_val = i2w_eng
                i2w_fre_val = i2w_fre
                for i in range(len(inp)):
                    inp[i, :] = torch.LongTensor( [w2i_eng_train[i2w_eng_val[w_i.item()]] if i2w_eng_val[w_i.item()] in w2i_eng_train  
                                     else w2i_eng_train['<unk>'] for w_i in list(inp[i, :])])

                for i in range(len(label)):
                    label[i, :] = torch.LongTensor([w2i_fre_train[i2w_fre_val[w_i.item()]] if i2w_fre_val[w_i.item()] in w2i_fre_train  
                                     else w2i_fre_train['<unk>'] for w_i in list(label[i, :])])

            elif status == 'test':
                i2w_eng_test = i2w_eng
                i2w_fre_test = i2w_fre
                for i in range(len(inp)):
                    inp[i, :] = torch.LongTensor( [w2i_eng_train[i2w_eng_test[w_i.item()]] if i2w_eng_test[w_i.item()] in w2i_eng_train  
                                     else w2i_eng_train['<unk>'] for w_i in list(inp[i, :])])

                for i in range(len(label)):
                    label[i, :] = torch.LongTensor([w2i_fre_train[i2w_fre_test[w_i.item()]] if i2w_fre_test[w_i.item()] in w2i_fre_train  
                                     else w2i_fre_train['<unk>'] for w_i in list(label[i, :])])

            inp = inp.to(device)
            label = label.to(device)

            output = model(inp, label)
            output = output[1:].view(-1, output.shape[-1])

            label = label.permute(1, 0)
            label = label[1:].reshape(-1)

            # print(output, label)

            loss = criterion(output, label)
            total_loss += loss.item()
        
        return total_loss/len(data_loader)

def train(model, epochs, batch_size, data_loader_train, data_loader_val, w2i_eng_train, i2w_eng_val,
     w2i_fre_train, i2w_fre_val, device): 

    optimizer = optim.Adam(model.parameters())
    criterion = nn.CrossEntropyLoss().to(device)
    best_loss = float('inf')

    for epoch in range(epochs):
        
        train_loss = 0
        model.train()

        for inp, label in tqdm(data_loader_train):
            inp = inp.to(device)
            label = label.to(device)
            
            optimizer.zero_grad()
            output = model(inp, label)
            
            output = output[1:].view(-1, output.shape[-1])
            label = label.permute(1, 0)
            label = label[1:].reshape(-1)
            
            try:
                loss = criterion(output, label)
            except:
                print(output.shape, label.shape)
            loss.backward()
            
            optimizer.step()
            train_loss += loss.item() 
        
        val_loss = val_test(model, data_loader_val, criterion, w2i_eng_train, i2w_eng_val, w2i_fre_train, i2w_fre_val, device, 'val')

        if val_loss < best_loss:
            best_loss = val_loss
            torch.save(model.state_dict(), 'attn_seq2seq.pth')
        print('Epoch : ', epoch)
        print('Train loss per-input : ', train_loss/len(data_loader_train))
        print('Val loss per-input : ', val_loss)