

def token_index(token_list):
    w2i = {tok : i for i, tok in enumerate(token_list)}
    i2w = dict([(value, key) for key, value in w2i.items()])
    return w2i, i2w

def tokenize_list(data, eng_lm, fre_lm):
    tokens_eng1 = list(dict.fromkeys([tok.text for tok in eng_lm.tokenizer(" ".join(i[0] for i in data))]))
    tokens_eng1.insert(0, '<sos>')
    tokens_eng1.insert(1, '<eos>')
    tokens_eng1.insert(2, '<unk>')
    tokens_fre1 = list(dict.fromkeys([tok.text for tok in fre_lm.tokenizer(" ".join(i[1] for i in data))]))
    tokens_fre1.insert(0, '<sos>')
    tokens_fre1.insert(1, '<eos>')
    tokens_fre1.insert(2, '<unk>')

    return tokens_eng1, tokens_fre1

def get_vocab(data, eng_lm, fre_lm):
	tokens_eng, tokens_fre = tokenize_list(data, eng_lm, fre_lm)
	# tokens_eng_val, tokens_fre_val = tokenize_list(val)
	# tokens_eng_test, tokens_fre_test = tokenize_list(test)

	w2i_eng, i2w_eng = token_index(tokens_eng)
	# w2i_eng_val, i2w_eng_val = token_index(tokens_eng_val)
	# w2i_eng_test, i2w_eng_test  = token_index(tokens_eng_test)

	w2i_fre, i2w_fre = token_index(tokens_fre)
	# w2i_fre_val, i2w_fre_val = token_index(tokens_fre_val)
	# w2i_fre_test, i2w_fre_test = token_index(tokens_fre_test)
	return w2i_eng, i2w_eng, w2i_fre, i2w_fre




