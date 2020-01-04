import torch
import torch.utils.data as data
from torch.utils.data.dataset import Dataset
from torch.utils.data import DataLoader

class Translation(Dataset):
    
    def __init__(self, eng_fre_data, eng_indeces, fre_indeces, eng_lm, fre_lm):

        #can either be train, val or test depending on the loader
        self.data = eng_fre_data 
        
        self.eng_indeces = eng_indeces
        self.fre_indeces = fre_indeces
        self.eng_lm = eng_lm
        self.fre_lm = fre_lm

    def __getitem__(self, index):
        
        input_tknzd = self.eng_lm.tokenizer(self.data[index][0])
        label_tknzd = self.fre_lm.tokenizer(self.data[index][1])
        
        input_indeces = [self.eng_indeces['<sos>']]    
        input_indeces[1:] = [self.eng_indeces[str(tok)] if str(tok) in 
                             self.eng_indeces else self.eng_indeces['<unk>'] for tok in input_tknzd]
        input_indeces.append(self.eng_indeces['<eos>'])
        
        label_indeces = [self.fre_indeces['<sos>']]
        label_indeces[1:] = [self.fre_indeces[str(tok)] if str(tok) in 
                             self.fre_indeces else self.fre_indeces['<unk>'] for tok in label_tknzd]
        label_indeces.append(self.fre_indeces['<eos>'])
        
        return torch.LongTensor(input_indeces), torch.LongTensor(label_indeces)
    
    def __len__(self):
        return len(self.data)
        

#  Dynamically varying batch shape conditioned on the max seqlen of an input in batch
def collate_fn(data):
    batch_inputs, batch_labels = zip(*data)
    
    inp_len = [len(inp) for inp in batch_inputs]
    label_len = [len(label) for label in batch_labels]
    
    inputs = torch.zeros((len(batch_inputs), max(inp_len)), dtype = torch.int64)
    labels = torch.zeros((len(batch_labels), max(label_len)), dtype = torch.int64)
    
    for i, inp in enumerate(batch_inputs):
        inputs[i, :len(inp)] = inp
    for i, label in enumerate(batch_labels):
        labels[i, :len(label)] = label
    
    return inputs, labels

def get_data_loader(data, w2i_eng, w2i_fre, batch_size, eng_lm, fre_lm):
    trans_data = Translation(data, w2i_eng, w2i_fre, eng_lm, fre_lm)
    data_loader = DataLoader(dataset = trans_data, batch_size = batch_size,
                                   shuffle = False,  collate_fn = collate_fn)
    return data_loader