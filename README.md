# Europeana BERT and ELECTRA models

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.4275044.svg)](https://doi.org/10.5281/zenodo.4275044)

In this repository we open source BERT and ELECTRA models trained on various Europeana newspapers.

Notice: We use the open source [Europeana newspapers](http://www.europeana-newspapers.eu/)
that were provided by *The European Library*. However, *The European Library* was permanently
shut down (end of 2019) and thus all direct download links are no longer working.
We are working on providing an alternative download mirror.

# Changelog

* 06.02.2020: Public release of German Europeana DistilBERT and ConvBERT models
* 16.11.2020: Public release of French Europeana BERT and ELECTRA models
* 26.07.2020: Public release of German Europeana ELECTRA model
* 10.02.2020: Initial version of this repo

# Model types

We trained different models, that are described in more detail in the following papers:

* [BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding](https://arxiv.org/abs/1810.04805)
* [ELECTRA: Pre-training Text Encoders as Discriminators Rather Than Generators](https://arxiv.org/abs/2003.10555)
* [ConvBERT: Improving BERT with Span-based Dynamic Convolution](https://arxiv.org/abs/2008.02496)
* [DistilBERT, a distilled version of BERT: smaller, faster, cheaper and lighter](https://arxiv.org/abs/1910.01108)

# German Europeana newspapers

We extracted all German texts using the `language` metadata attribute from the Europeana corpus.

## Stats

The resulting corpus has a size of 51GB and consists of 8,035,986,369 tokens.

Based on the metadata information, texts from the 18th - 20th century are mainly included in the
training corpus. The following figure shows a detailed overview (tokens per year):

![Tokens per year for German Europeana training corpus](german_year_token_stats.png)

## Pretraining

We use the awesome 🤗 / Tokenizers library for building the BERT-compatible vocab (32,000 subwords).

We use the same preprocessing steps and training parameters as for our Turkish BERT and ELECTRA models.
A cheatsheet for can be found [here for BERT](https://github.com/stefan-it/turkish-bert/blob/master/CHEATSHEET.md),
and [here for ELECTRA](https://github.com/stefan-it/turkish-bert/blob/master/electra/CHEATSHEET.md).

The ConvBERT model was trained with the [reference implementation](https://github.com/yitu-opensource/ConvBert)
on a v3-8 TPU with a maximum sequence length of 512 over the complete training corpus for 1M steps.

For the DistilBERT model, only 6GB of the original training data (51GB) was used. The model was trained for
two epochs with the [reference implementation](https://github.com/huggingface/transformers/tree/master/examples/research_projects/distillation)
from Transformers on 4 RTX 3090 with a batch size of 6.

## Results

We mainly compare our models against the NER dataset, that is used in the following
two papers:

* [A Named Entity Recognition Shootout for German](https://www.aclweb.org/anthology/P18-2020/)
  by Riedl and Padó
* [Towards Robust Named Entity Recognition for Historic German](https://www.aclweb.org/anthology/W19-4312/)
  by Schweter and Baiter

The datasets used in these papers are: ONB (Austrian National Library) and LFT (Dr Friedrich Teman Library).
LFT corpus consists of texts from 1926, ONB corpus consists of texts ranging from 1710 - 1873.

We include various other BERT models (incl. XLM-R) in our comparison. Note: Schweter and Baiter used Flair
Embeddings (stacked with Word Embeddings) resulting in the current SOTA for these tasks.

We use the awesome Flair library for experiments with our Transformer-based models.
It nicely wraps the outstanding 🤗 Transformers library and we can also use the
HuggingFace model hub. For training our NER models we use the `flair-ner-trainer.py` script and `flair-ner-predictor.py`
for predicting and piping the output to the CoNLL-2003 evaluation script.

Please star and watch [Flair](https://github.com/flairNLP/flair) and [Transformers](https://github.com/huggingface/transformers)
on GitHub!

### LFT (1926)

We measure averaged F-score over 5 runs using the official [CoNLL-2003 evaluation script](https://www.clips.uantwerpen.be/conll2003/ner/bin/conlleval).
Results on the development set in brackets:

| Model                                     | Run 1           | Run 2           | Run 3           | Run 4           | Run 5           | Avg.
| ----------------------------------------- | --------------- | --------------- | --------------- | --------------- | --------------- | --------------------------
| mBERT (base, cased)                       | (74.99) / 76.71 | (75.65) / 77.02 | (76.65) / 78.04 | (74.93) / 78.15 | (75.52) / 77.61 | (75.55) / 77.51 ± 0.56
| German DBMDZ Bert (base, cased)           | (75.61) / 77.14 | (76.79) / 78.74 | (75.42) / 77.70 | (76.23) / 77.61 | (75.58) / 78.07 | (78.93) / 77.85 ± 0.53
| XLM-R (large, cased)                      | (77.04) / 76.35 | (76.03) / 74.80 | (77.62) / 75.23 | (76.60) / 74.87 | (76.78) / 76.11 | (76.81) / 75.48 ± 0.64
| German Europeana BERT (base, cased)       | (78.75) / 78.81 | (78.59) / 79.88 | (79.28) / 78.87 | (79.55) / 78.15 | (79.42) / 78.20 | (79.12) / 78.78 ± 0.62
| German Europeana BERT (base, uncased)     | (76.96) / 76.83 | (76.16) / 74.62 | (75.63) / 74.77 | (76.53) / 75.83 | (76.47) / 77.07 | (76.35) / 75.82 ± 1.04
| German Europeana ELECTRA (base, cased)    | (80.76) / 81.86 | (80.42) / 81.12 | (80.35) / 81.28 | (80.37) / 81.49 | (79.95) / 80.72 | (80.37) / 81.26 ± 0.38
| German Europeana DistilBERT (base, cased) | (77.87) / 77.69 | (77.75) / 77.33 | (77.70) / 77.46 | (78.63) / 77.56 | (78.10) / 78.60 | (78.01) / 77.73 ± 0.45
| German Europeana ELECTRA (large, cased)   | (81.89) / 83.41 | (81.49) / 82.61 | (81.52) / 83.48 | (81.42) / 81.99 | (81.84) / 82.53 | (81.63) / **82.80** ± 0.57
| German Europeana ConvBERT (base, cased)   | (81.06) / 82.06 | (81.15) / 81.66 | (80.69) / 81.77 | (81.61) / 81.74 | (81.06) / 81.55 | (81.11) / 81.76 ± 0.17

The current SOTA reported by Schweter and Baiter is 77.51%.

### ONB (1710 - 1873)

We measure averaged F-score over 5 runs using the official [CoNLL-2003 evaluation script](https://www.clips.uantwerpen.be/conll2003/ner/bin/conlleval).
Results on the development set in brackets:

| Model                                     | Run 1           | Run 2           | Run 3           | Run 4           | Run 5           | Avg.
| ----------------------------------------- | --------------- | --------------- | --------------- | --------------- | --------------- | --------------------------
| mBERT (base, cased)                       | (82.06) / 80.81 | (82.24) / 77.61 | (82.64) / 81.95 | (83.21) / 83.38 | (82.25) / 82.07 | (82.48) / 81.16 ± 1.95
| German DBMDZ Bert (base, cased)           | (84.44) / 84.35 | (84.70) / 83.49 | (84.26) / 82.34 | (85.64) / 80.47 | (85.19) / 83.79 | (84.85) / 82.89 ± 1.38
| XLM-R (large, cased)                      | (82.16) / 81.64 | (82.45) / 80.64 | (83.70) / 81.63 | (82.68) / 80.85 | (81.76) / 80.74 | (82.55) / 81.10 ± 0.44
| German Europeana BERT (base, cased)       | (88.60) / 85.39 | (87.77) / 85.42 | (88.76) / 86.41 | (88.36) / 86.04 | (88.49) / 85.86 | (88.40) / 85.82 ± 0.39
| German Europeana BERT (base, uncased)     | (86.74) / 84.18 | (86.84) / 82.88 | (86.19) / 83.89 | (86.87) / 84.54 | (86.50) / 84.72 | (86.63) / 84.04 ± 0.65
| German Europeana ELECTRA (base, cased)    | (88.09) / 86.75 | (87.72) / 85.35 | (86.86) / 84.63 | (87.75) / 86.38 | (87.93) / 86.05 | (87.67) / 85.83 ± 0.76
| German Europeana DistilBERT (base, cased) | (89.29) / 85.11 | (87.86) / 85.68 | (87.68) / 86.29 | (87.22) / 86.65 | (88.04) / 87.74 | (88.02) / 86.29 ± 0.89
| German Europeana ELECTRA (large, cased)   | (88.84) / 86.45 | (88.89) / 88.14 | (88.49) / 87.42 | (89.99) / 85.94 | (88.57) / 86.52 | (88.96) / **86.89** ± 0.78
| German Europeana ConvBERT (base, cased)   | (89.31) / 87.89 | (89.18) / 85.71 | (88.97) / 86.16 | (90.10) / 85.75 | (87.80) / 85.86 | (89.07) / 86.27 ± 0.82

The current SOTA reported by Schweter and Baiter is 85.31%.

Notice: The ONB dataset covers texts from 1710 - 1873. The Europeana training dataset only contains little data for this specific
time period. We are currently working on BERT models with other training data for this period!

## Model weights

Most of the pre-trained models are located on the 🤗 / model hub:

* [German Europeana BERT (base, cased)](https://huggingface.co/dbmdz/bert-base-german-europeana-cased)
* [German Europeana BERT (base, uncased)](https://huggingface.co/dbmdz/bert-base-german-europeana-uncased)
* [German Europeana ELECTRA (base, cased, discriminator)](https://huggingface.co/dbmdz/electra-base-german-europeana-cased-discriminator)
* [German Europeana ELECTRA (base, cased, generator)](https://huggingface.co/dbmdz/electra-base-german-europeana-cased-generator)
* [German Europeana ConvBERT (base. cased)](https://huggingface.co/dbmdz/convbert-base-german-europeana-cased)
* [German Europeana DistilBERT (base, cased)](https://huggingface.co/dbmdz/distilbert-base-german-europeana-cased)

# French Europeana newspapers

We also extracted all French texts using the `language` metadata attribute from the Europeana corpus.

## Stats

The resulting corpus has a size of 63GB and consists of 11,052,528,456 tokens.

Based on the metadata information, texts from the 18th - 20th century are mainly included in the
training corpus. The following figure shows a detailed overview (tokens per year):

![Tokens per year for French Europeana training corpus](french_year_token_stats.png)

## Pretraining

We use the awesome 🤗 / Tokenizers library for building the BERT-compatible vocab (32,000 subwords).

We use the same preprocessing steps and training parameters as for our Turkish BERT and ELECTRA models.
A cheatsheet for can be found [here for BERT](https://github.com/stefan-it/turkish-bert/blob/master/CHEATSHEET.md),
and [here for ELECTRA](https://github.com/stefan-it/turkish-bert/blob/master/electra/CHEATSHEET.md).

## Model weights

Both BERT and ELECTRA model weights for PyTorch and TensorFlow are available.

* French Europeana BERT: `dbmdz/bert-base-french-europeana-cased` - [model hub page](https://huggingface.co/dbmdz/bert-base-french-europeana-cased/tree/main)
* French Europeana ELECTRA (discriminator): `dbmdz/electra-base-french-europeana-cased-discriminator` - [model hub page](https://huggingface.co/dbmdz/electra-base-french-europeana-cased-discriminator/tree/main)
* French Europeana ELECTRA (generator): `dbmdz/electra-base-french-europeana-cased-generator` - [model hub page](https://huggingface.co/dbmdz/electra-base-french-europeana-cased-generator/tree/main)

# License

All models are licensed under [MIT](LICENSE).

# Huggingface model hub

All models are available on the [Huggingface model hub](https://huggingface.co/dbmdz).

# Contact (Bugs, Feedback, Contribution and more)

For questions about our Europeana BERT, ELECTRA and ConvBERT models just open a new discussion
[here](https://github.com/stefan-it/europeana-bert/discussions) 🤗

# Citation

You can use the following BibTeX entry for citation:

```bibtex
@software{stefan_schweter_2020_4275044,
  author       = {Stefan Schweter},
  title        = {Europeana BERT and ELECTRA models},
  month        = nov,
  year         = 2020,
  publisher    = {Zenodo},
  version      = {1.0.0},
  doi          = {10.5281/zenodo.4275044},
  url          = {https://doi.org/10.5281/zenodo.4275044}
}
```

# Acknowledgments

Research supported with Cloud TPUs from Google's TensorFlow Research Cloud (TFRC).
Thanks for providing access to the TFRC ❤️

Thanks to the generous support from the [Hugging Face](https://huggingface.co/) team,
it is possible to download both cased and uncased models from their S3 storage 🤗
