# STEPS FOR TRAINING

First download the two spacy language models of english and french. This will be leveraged for tokenization. 

`python3 -m spacy download fr`

`python3 -m spacy download en`

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
