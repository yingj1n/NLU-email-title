# NLU-email-title
  Informal and noisy text summarization is a meaningful task with practical use potentials. In this project, we adopted the data-driven attention-based neural model on the task to generate email titles. 
    The experiment evaluations show that our model achieves 21.22 Rouge-L and 17.10 BLEU. We showed that the attention-based model outperforms the extractive methods, though informal text summarization remains to be a challenge.
## Datasets

The Enron Corpus were used in this project. It is parsed by the `Parsing Enron Data.ipynb` under the notebook folder. The parsing outputs a CSV file that contains subject line and first email body. 

## Baseline Model

The baseline model is TextRank, which is a extractive graph-based model. The notebook `baseline_test.ipynb` demonstrated the process of extracting subject lines and computing rouge scores. The extractive technique does not require further training.

## Encoder-Decoder Model

This is an abstractibe model, also known as ABS. We use notebook `Training Encoder Decoder.ipynb` to train the model. Our encoder is a bidirectional GRU (gated recurrent unit) and the decoder is attentional based. The training takes roughly 4 hours on the entire preprocessed enron corpus.
