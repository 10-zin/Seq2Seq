import re



def split(path):

	with open(path, encoding='utf-8') as f:
	    content = f.readlines()

	eng_fre_pairs = []
	for line in content:
	    line = re.sub ('\u202f', '', line)
	    line = line.strip('\n').split('\t')
	    eng_fre_pairs.append(line)


	train = []
	val = []
	test = []
	for i, pair in enumerate(eng_fre_pairs):
	    if i%5 == 0:
	        test.append(pair)
	    elif i%17 == 0:
	        val.append(pair)
	    else:
	        train.append(pair)

	return train, val, test