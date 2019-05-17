# NLU-email-title
some github sources

attention based summarization
https://github.com/facebookarchive/NAMAS

info retriv
https://github.com/gr33ndata/irlib

## Datasets

The Enron Corpus were used in this project. It is parsed by the `Parsing Enron Data.ipynb` under the notebook folder. The parsing outputs a CSV file that contains subject line and first email body. 

## Baseline Model

The baseline model is TextRank, which is a extractive graph-based model. The notebook `baseline_test.ipynb` demonstrated the process of extracting subject lines and computing rouge scores. The extractive technique does not require further training.

## Encoder-Decoder Model

This is an abstractibe model, also known as ABS. We use notebook `Training Encoder Decoder.ipynb` to train the model. Our encoder is a bidirectional GRU (gated recurrent unit) and the decoder is attentional based. The training takes roughly 4 hours on the entire preprocessed enron corpus.
