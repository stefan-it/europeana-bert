# Europeana BERT and ELECTRA models

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.4275044.svg)](https://doi.org/10.5281/zenodo.4275044)

In this repository we open source BERT and ELECTRA models trained on various Europeana newspapers.

Notice: We use the open source [Europeana newspapers](http://www.europeana-newspapers.eu/)
that were provided by *The European Library*. However, *The European Library* was permanently
shut down (end of 2019) and thus all direct download links are no longer working.
We are working on providing an alternative download mirror.

# Changelog

* 25.07.2021: Add fine-tuned evaluations results for various models
* 06.02.2021: Public release of German Europeana DistilBERT and ConvBERT models
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
HuggingFace model hub. 

We evaluate both feature-based and fine-tuned NER models. For the feature-based approach we use the mean over all layers from the
Transformer model. We use an initial learning rate of 0.1. Then we reduce the learning rate by a factor of 0.5 with a patience of
3. This factor determines the number of epochs with no improvement after which learning rate will be reduced. We use a batch size of 16
for training feature-based models and use different seeds for training 5 models. Averaged F1-score is then calculated.

For the fine-tuned models we perform a hyper-parameter search over:

* `batch_size`: [8, 16]
* `epochs`: [1, 5, 10]
* `learning_rates`: [1e-5, 3e-5, 5e-5]

Then we choose the best hyper-parameter configuration and train 5 models with different seeds and average F1-score over these models.

For training our feature-based NER models we use the `flair-ner-trainer.py` script and `flair-ner-predictor.py`
for predicting and piping the output to the CoNLL-2003 evaluation script. We use `flair-ner-trainer-ft.py` for fine-tuning, with all
necessary configuration files located in the `configs` folder of this repository.

Please star and watch [Flair](https://github.com/flairNLP/flair) and [Transformers](https://github.com/huggingface/transformers)
on GitHub!

### LFT (1926)

#### Feature-based evaluation

| Model                    | Development F1-score | Test F1-score
| ------------------------ | -------------------- | -----------------
| DBMDZ German BERT        | 75.926 ± 0.57        | 77.852 ± 0.60
| Europeana BERT (cased)   | 79.118 ± 0.42        | 78.782 ± 0.70
| Europeana BERT (uncased) | 76.350 ± 0.49        | 75.824 ± 1.13
| Europeana ConvBERT       | **81.114** ± 0.33    | **81.756** ± 0.19
| Europeana DistilBERT     | 78.010 ± 0.38        | 77.728 ± 0.51
| Europeana ELECTRA        | 80.370 ± 0.29        | 81.294 ± 0.42
| XLM-R (base)             | 76.814 ± 0.58        | 75.472 ± 0.72
| mBERT (cased)            | 75.548 ± 0.69        | 77.506 ± 0.63

![LFT Feature-based Development Results](figures/lft_fb_dev.png)
![LFT Feature-based Test Results](figures/lft_fb_test.png)

#### Fine-based evaluation

| Model                    | Development F1-score | Test F1-score
| ------------------------ | -------------------- | -----------------
| DBMDZ German BERT        | 77.336 ± 0.53        | 77.544 ± 0.72
| Europeana BERT (cased)   | 79.278 ± 0.60        | 79.734 ± 0.85
| Europeana BERT (uncased) | 76.938 ± 0.53        | 77.984 ± 0.97
| Europeana ConvBERT       | **80.550** ± 0.44    | **81.352** ± 1.54
| Europeana DistilBERT     | 79.086 ± 0.54        | 77.606 ± 1.31
| Europeana ELECTRA        | 79.914 ± 0.87        | 81.062 ± 0.80
| GBERT (base)             | 76.984 ± 0.42        | 76.826 ± 0.47
| XLM-R (base)             | 76.142 ± 0.46        | 75.898 ± 0.55
| mBERT (cased)            | 77.804 ± 0.66        | 78.362 ± 0.74

![LFT Fine-tuned Development Results](figures/lft_ft_dev.png)
![LFT Fine-tuned Test Results](figures/lft_ft_test.png)

The current SOTA reported by Schweter and Baiter is 77.51% on test set.

### ONB (1710 - 1873)

#### Feature-based evaluation

| Model                    | Development F1-score | Test F1-score
| ------------------------ | -------------------- | -----------------
| DBMDZ German BERT        | 84.846 ± 0.57        | 82.888 ± 1.54
| Europeana BERT (cased)   | 88.396 ± 0.38        | 85.824 ± 0.43
| Europeana BERT (uncased) | 86.628 ± 0.28        | 84.042 ± 0.72
| Europeana ConvBERT       | **89.072** ± 0.83    | 86.274 ± 0.92
| Europeana DistilBERT     | 88.018 ± 0.77        | **86.294** ± 1.00
| Europeana ELECTRA        | 87.670 ± 0.48        | 85.832 ± 0.85
| XLM-R (base)             | 82.550 ± 0.73        | 81.100 ± 0.49
| mBERT (cased)            | 82.480 ± 0.46        | 81.164 ± 2.19

![ONB Feature-based Development Results](figures/onb_fb_dev.png)
![ONB Feature-based Test Results](figures/onb_fb_test.png)

#### Fine-based evaluation

| Model                    | Development F1-score | Test F1-score
| ------------------------ | -------------------- | -----------------
| DBMDZ German BERT        | 84.490 ± 0.57        | 82.966 ± 0.56
| Europeana BERT (cased)   | 87.198 ± 0.40        | 85.282 ± 0.82
| Europeana BERT (uncased) | 87.004 ± 0.52        | 85.228 ± 0.90
| Europeana ConvBERT       | **88.652** ± 0.75    | **87.284** ± 0.64
| Europeana DistilBERT     | 85.004 ± 1.08        | 84.682 ± 0.65
| Europeana ELECTRA        | 87.856 ± 0.76        | 86.064 ± 0.55
| GBERT (base)             | 84.014 ± 1.08        | 83.004 ± 0.87
| XLM-R (base)             | 83.942 ± 1.10        | 82.056 ± 0.84
| mBERT (cased)            | 84.360 ± 0.71        | 83.064 ± 0.80

![ONB Fine-tuned Development Results](figures/onb_ft_dev.png)
![ONB Fine-tuned Test Results](figures/onb_ft_test.png)

The current SOTA reported by Schweter and Baiter is 85.31% on test set.

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
