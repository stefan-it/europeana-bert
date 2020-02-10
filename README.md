# Europeana BERT models

In this repository we open source BERT models trained on various Europeana newspapers.

Notice: We use the open source [Europeana newspapers](http://www.europeana-newspapers.eu/)
that we provided by *The European Library*. However, the *The European Library* was permanently
shut down (end of 2019) and thus all direct download links are no longer working.
We are working on providing an alternative download mirror.

# Changelog

* 10.02.2020: Initial version of this repo

# German Europeana newspapers

We extracted all German texts using the `language` metadata attribute from the Europeana corpus.


## Stats

The resulting corpus has a size of 51GB and consists of 8,035,986,369 tokens.

Based on the metadata information, texts from the 18th - 20th century are mainly included in the
training corpus. The following figure shows a detailed overview (tokens per year):

![Tokens per year for German Europeana training corpus](german_year_token_stats.png)

## Pretraining

We the awesome 🤗 / Tokenizers library for building the BERT-compatible vocab (32,000 subwords).

We use the same preprocessing steps and training parameters as for our (upcoming) Turkish BERT model.
A cheatsheet can be found [here](https://github.com/stefan-it/turkish-bert/blob/master/CHEATSHEET.md).

## Results

We mainly compare our cased and uncased models against the NER dataset, that is used in the following
two papers:
Padó
* [A Named Entity Recognition Shootout for German](https://www.aclweb.org/anthology/P18-2020/)
  by Riedl and Padó
* [Towards Robust Named Entity Recognition for Historic German](https://www.aclweb.org/anthology/W19-4312/)
  by Schweter and Baiter

The datasets used in these papers are: ONB (Austrian National Library) and LFT (Dr Friedrich Teman Library).
LFT corpus consists of texts from 1926, ONB corpus consists of texts ranging from 1710 - 1873.

We include various other BERT models (incl. XLM-R) in our comparison. Note: Schweter and Baiter used Flair
Embeddings (stacked with Word Embeddings) resulting in the current SOTA for these tasks.

We use the `run_ner.py` script from the awesome 🤗 / Transformers library with default parameters and a
batch size of `16`.

### LFT (1926)

We measure averaged F-score over 5 runs. Results on the development set in brackets:

| Model                                 | Run 1           | Run 2           | Run 3           | Run 4           | Run 5           | Avg.
| ------------------------------------- | --------------- | --------------- | --------------- | --------------- | --------------- | ------------------
| mBERT (base, cased)                   | (75.40) / 76.76 | (76.57) / 77.89 | (75.93) / 76.81 | (75.87) / 77.35 | (75.63) / 77.48 | (75.88) / 77.26
| German DBMDZ Bert (base, cased)       | (76.95) / 75.79 | (76.32) / 76.28 | (76.68) / 76.81 | (75.95) / 77.69 | (76.17) / 75.77 | (76.41) / 76.47
| XLM-R (large, cased)                  | (76.88) / 76.73 | (78.99) / 77.71 | (78.82) / 76.42 | (77.48) / 77.98 | (78.75) / 77.45 | (78.18) / 77.26
| German Europeana BERT (base, cased)   | (79.06) / 80.90 | (78.91) / 79.55 | (80.46) / 79.53 | (79.10) / 81.11 | (79.25) / 81.67 | (79.36) / **80.55**
| German Europeana BERT (base, uncased) | (76.93) / 77.66 | (76.01) / 78.08 | (76.53) / 78.05 | (77.29) / 77.48 | (76.54) / 78.36 | (76.66) / 77.93

The current SOTA reported by Schweter and Baiter is 77.51%. Thus, our trained BERT model is ~3% better.

### ONB (1710 - 1873)

We measure averaged F-score over 5 runs. Results on the development set in brackets:

| Model                                 | Run 1           | Run 2           | Run 3           | Run 4           | Run 5           | Avg.
| ------------------------------------- | --------------- | --------------- | --------------- | --------------- | --------------- | ------------------
| mBERT (base, cased)                   | (84.25) / 83.80 | (83.67) / 85.20 | (82.96) / 82.02 | (84.62) / 84.26 | (85.54) / 81.90 | (84.21) / 83.44
| German DBMDZ Bert (base, cased)       | (83.92) / 83.40 | (85.30) / 83.02 | (83.81) / 83.46 | (84.25) / 83.44 | (84.42) / 83.73 | (84.34) / 83.41
| XLM-R (large, cased)                  | (86.70) / 86.00 | (86.25) / 86.60 | (84.77) / 86.00 | (88.38) / 84.57 | (86.09) / 85.64 | (86.44) / **85.76**
| German Europeana BERT (base, cased)   | (86.63) / 85.92 | (88.11) / 85.25 | (88.20) / 84.26 | (87.38) / 87.45 | (86.49) / 84.60 | (87.36) / 85.50
| German Europeana BERT (base, uncased) | (87.95) / 84.74 | (87.02) / 85.68 | (88.34) / 84.42 | (88.18) / 86.14 | (87.28) / 85.03 | (87.75) / 85.20

The current SOTA reported by Schweter and Baiter is 85.31%. Both XLM-R and German Europeana BERT are slightly better.

Notice: The ONB dataset covers texts from 1710 - 1873. The Europeana training dataset only contains little data for this specific
time period. We are currently working on BERT models with other training data for this period!

## Usage

Both cased and uncased models are located on the 🤗 / model hub and can be loaded like:

```python
from transformers import AutoModel, AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained("dbmdz/bert-base-german-europeana-cased")
model = AutoModel.from_pretrained("dbmdz/bert-base-german-europeana-cased")
```

The uncased model can be loaded like:

```python
from transformers import AutoModel, AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained("dbmdz/bert-base-german-europeana-uncased")
model = AutoModel.from_pretrained("dbmdz/bert-base-german-europeana-uncased")
```

# Huggingface model hub

All models are available on the [Huggingface model hub](https://huggingface.co/dbmdz).

# Contact (Bugs, Feedback, Contribution and more)

For questions about our Europeana BERT models just open an issue
[here](https://github.com/stefan-it/europeana-bert/issues) 🤗

# Acknowledgments

Research supported with Cloud TPUs from Google's TensorFlow Research Cloud (TFRC).
Thanks for providing access to the TFRC ❤️

Thanks to the generous support from the [Hugging Face](https://huggingface.co/) team,
it is possible to download both cased and uncased models from their S3 storage 🤗
