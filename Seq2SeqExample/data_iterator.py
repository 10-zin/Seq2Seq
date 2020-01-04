import spacy
from torchtext.data import Field, BucketIterator
from torchtext.datasets import Multi30k

spacy_de = spacy.load('de_core_news_sm')
spacy_en = spacy.load('en')

def tokenize_de(text):
    """
    Tokenizes German text from a string into a list of strings (tokens) and reverses it
    """
    return [tok.text for tok in spacy_de.tokenizer(text)]

def tokenize_en(text):
    """
    Tokenizes English text from a string into a list of strings (tokens)
    """
    return [tok.text for tok in spacy_en.tokenizer(text)]

def get_data_iterator(batch_size, device):

	SRC = Field(tokenize = tokenize_de,
           init_token = '<sos>',
           eos_token = '<eos>',
           lower = True)
	TRG = Field(tokenize = tokenize_en,
            init_token = '<sos>',
            eos_token = '<eos',
            lower = True)

	train_data, valid_data, test_data = Multi30k.splits(exts = ('.de', '.en'), fields = (SRC, TRG)) 
	SRC.build_vocab(train_data, min_freq = 2)
	TRG.build_vocab(train_data, min_freq = 2)

	train_iterator, valid_iterator, test_iterator = BucketIterator.splits(
	    (train_data, valid_data, test_data),
	    batch_size = batch_size,
	    device = device)

	return train_iterator, valid_iterator, test_iterator, SRC, TRG
