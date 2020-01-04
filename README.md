# Seq2Seq

A collection of various types of seq2seq models.
A Sequence to Sequence model takes a sequence from one domain as an input and outputs a sequence from another domain as an output. Technically, It uses an encoder-decoder architecture to encode a given input sequence to its context representation and then decode it to an outpt sequence.

More specifically, we find its use in machine translation. Where, the model can learn to translate an input from a given language to a target language.

Each encoder and decoder, is nothing but a recurrent network, a LSTM or GRU in this repository. 

Here, I implement three types of sequence to sequence models with increasing complexity and performance.
1) Basic seq2seq - Multi-layer LSTM based Encoder-Decoder.
2) Seq2SeqPR - Context Vector dependent GRU based Ecoder-Decoder.
3) Seq2SeqAttention - Attention dependent GRU based Encoder-Decoder. 

Each directory includes the script for the respective implementation.




