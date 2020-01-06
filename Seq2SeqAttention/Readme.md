# Documentation

First download the two spacy language models of english and french. This will be leveraged for tokenization. 

`python3 -m spacy download fr`

`python3 -m spacy download en`

Next, download the machine translation data and store it in the data directory. Refer to the data link provided in \data of this repo.

Next, run the following command to train the model with default arguments. 

`python3 main.py data/eng-fra.txt`

If you would like to change the default arguments check the available options from the following command.

`python3 main.py -h`

It will show the following arguments.

Positional Arguments

| Argument | Description |
|----------|-------------|
|path      |Enter the path to data|

Optional Arguments

| Argument   | Description                        |  Default |
|------------|------------------------------------|----------|
| enc_hid    |Enter the encoder's hidden dimension|512       |
| dec_hid    |Enter the decoder's hidden dimension|512       |
| emb_dim    |Enter the embedding dimension       |300       |
| drop_prob  |Enter the droput probability        |0.5       |
| batch_size |Enter the batch size                |64        |
|  epochs    |Enter the number of epochs          |2         |
