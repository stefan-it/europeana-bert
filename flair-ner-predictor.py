import click
import logging
import sys

from typing import List

from flair import set_seed

from flair.data import Sentence
from flair.datasets import ColumnCorpus
from flair.embeddings import (
    TokenEmbeddings,
    StackedEmbeddings,
    TransformerWordEmbeddings
)
from flair.models import SequenceTagger
from flair.trainers import ModelTrainer


@click.command()
@click.option("--data_folder", required=True, type=str, help="Should point to ./data/")
@click.option("--task_name", required=True, type=str, help="Should be lft or onb")
@click.option("--model_name", required=True, type=str, help="Should be best Flair like best-model.pt")
@click.option("--split", required=True, type=str, help="Should be dev or test")
def run_experiment(data_folder, task_name, model_name, split):
    # Adjust logging level
    logging.getLogger("flair").setLevel(level="ERROR")

    set_seed(1)

    if task_name in ["lft", "onb"]:
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

    tagger: SequenceTagger = SequenceTagger.load(model_name)

    ds = corpus.test if split == "test" else corpus.dev

    for sentence in ds:
        tagger.predict(sentence, label_name="predicted")

        gold_spans = sentence.get_spans("ner")

        pred_spans = sentence.get_spans("predicted")

        for token in sentence:
            gold_tag = "O"

            for span in gold_spans:
                if token in span:
                    gold_tag = "B-" + span.tag if token == span[0] else "I-" + span.tag

            pred_tag = "O"

            for span in pred_spans:
                if token in span:
                    pred_tag = "B-" + span.tag if token == span[0] else "I-" + span.tag

            print(f"{token.text} {gold_tag} {pred_tag}")

        print("")

if __name__ == "__main__":
    run_experiment()  # pylint: disable=no-value-for-parameter
