import importlib 
import model, data_preprocess, vocab, data_loader, train_eval
import argparse
import os 
import spacy
import torch
import torch.nn as nn
# Download
# ! python -m spacy download fr
# ! python -m spacy download en

if __name__ == '__main__':

	parser = argparse.ArgumentParser()
	parser.add_argument('path', help = "Enter the path to data")
	parser.add_argument('--enc_hid', help = "Enter the encoder's hidden dimension", type = int, default = 512)
	parser.add_argument('--dec_hid', help = "Enter the decoder's hidden dimension", type = int, default = 512)
	parser.add_argument('--emb_dim', help = "Enter the embedding dimension", type = int, default = 300)
	parser.add_argument('--drop_prob', help = "Enter the dropout probability", type = float, default = 0.5)
	parser.add_argument('--batch_size', help = "Enter the batch size", type = int, default = 64)
	parser.add_argument('--epochs', help = "Enter the number of epochs", type = int, default = 2)
	args = parser.parse_args()
	# path = 'data/eng-fra.txt'


	train, val, test = data_preprocess.split(args.path)
	eng_lm = spacy.load('en')
	fre_lm = spacy.load('fr')

	w2i_eng_train, _, w2i_fre_train, _ = vocab.get_vocab(train, eng_lm, fre_lm)
	w2i_eng_val, i2w_eng_val, w2i_fre_val, i2w_fre_val = vocab.get_vocab(val, eng_lm, fre_lm)
	w2i_eng_test, i2w_eng_test, w2i_fre_test, i2w_fre_test = vocab.get_vocab(test, eng_lm, fre_lm)

	device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
	inp_vocab_dim = len(w2i_eng_train)
	label_vocab_dim = len(w2i_fre_train)

	m = model.enc_dec_attn(args.enc_hid, args.dec_hid, args.emb_dim, args.drop_prob, device, inp_vocab_dim, label_vocab_dim)
	# print(m)
	# print(f'The model has {model.count_parameters(m):,} trainable parameters')

	data_loader_train = data_loader.get_data_loader(train, w2i_eng_train, w2i_fre_train, args.batch_size, eng_lm, fre_lm)
	data_loader_val = data_loader.get_data_loader(val, w2i_eng_val, w2i_fre_val, args.batch_size, eng_lm, fre_lm)
	data_loader_test = data_loader.get_data_loader(test, w2i_eng_test, w2i_fre_test, args.batch_size, eng_lm, fre_lm)

	loss = train_eval.train(m, args.epochs, args.batch_size, data_loader_train, data_loader_val, w2i_eng_train, i2w_eng_val,
	 w2i_fre_train, i2w_fre_val, device)

	criterion = nn.CrossEntropyLoss().to(device)
	loss = train_eval.val_test(m, data_loader_test, criterion, w2i_eng_train, i2w_eng_test,
     w2i_fre_train, i2w_fre_test, device, 'test')

	print('Test Loss: {}'.format(loss))









