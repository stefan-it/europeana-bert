import click
import logging
import sys

from typing import List

from flair.data import Sentence
from flair.datasets import ColumnCorpus
from flair.embeddings import (
    TokenEmbeddings,
    StackedEmbeddings,
    TransformerWordEmbeddings
)
from flair.models import SequenceTagger
from flair.trainers import ModelTrainer



# Convert IOBES to IOB for CoNLL evaluations script
def iobes_to_iob(tag):
    iob_tag = tag

    if tag.startswith("S-"):
        iob_tag = tag.replace("S-", "B-")

    if tag.startswith("E-"):
        iob_tag = tag.replace("E-", "I-")

    return iob_tag

@click.command()
@click.option("--data_folder", required=True, type=str, help="Should point to ./data/")
@click.option("--task_name", required=True, type=str, help="Should be lft or onb")
@click.option("--model_name", required=True, type=str, help="Should be best Flair like best-model.pt")
@click.option("--split", required=True, type=str, help="Should be dev or test")
def run_experiment(data_folder, task_name, model_name, split):
    # Adjust logging level
    logging.getLogger("flair").setLevel(level="ERROR")

    # Configuration
    column_format = {0: "token", 1: "ner"}

    # We use official data from Riedl and Padó
    train_file = f"enp_DE.{task_name}.mr.tok.train.bio"
    dev_file = f"enp_DE.{task_name}.mr.tok.dev.bio"
    test_file = f"enp_DE.{task_name}.mr.tok.test.bio"

    # Corpus
    corpus = ColumnCorpus(data_folder=data_folder,
                          column_format=column_format,
                          train_file=train_file,
                          dev_file=dev_file,
                          test_file=test_file,
                          tag_to_bioes="ner",
                          )

    # Corpus configuration
    tag_type = "ner"
    tag_dictionary = corpus.make_tag_dictionary(tag_type=tag_type)

    tagger: SequenceTagger = SequenceTagger.load(model_name)
   
    ds = corpus.test if split == "test" else corpus.dev

    for sentence in ds:
        tokens = sentence.tokens
        gold_labels = [token.get_tag('ner').value for token in sentence.tokens]

        tagged_sentence = Sentence()
        tagged_sentence.tokens = tokens

        tagger.predict(tagged_sentence)

        predicted_labels = [token.get_tag('ner').value for token in tagged_sentence.tokens]

        assert len(tokens) == len(gold_labels)
        assert len(gold_labels) == len(predicted_labels)

        for token, gold_label, predicted_label in zip(tokens, gold_labels, predicted_labels):
            gold_label = iobes_to_iob(gold_label)
            predicted_label = iobes_to_iob(predicted_label)

            print(f"{token.text} {gold_label} {predicted_label}")
        
        print("")

if __name__ == "__main__":
    run_experiment()  # pylint: disable=no-value-for-parameter
