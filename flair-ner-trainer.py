import click
import sys

from typing import List

from flair.datasets import ColumnCorpus
from flair.embeddings import (
    TokenEmbeddings,
    StackedEmbeddings,
    TransformerWordEmbeddings
)
from flair import set_seed
from flair.models import SequenceTagger
from flair.trainers import ModelTrainer

@click.command()
@click.option("--data_folder", required=True, type=str, help="Should point to ./data/")
@click.option("--task_name", required=True, type=str, help="Should be lft or onb")
@click.option("--model_name", required=True, type=str, help="Should be a valid HuggingFace model hub name")
@click.option("--run_id", required=True, type=str, help="Should be [1-5]")
@click.option("--use_context", default=0, type=int, help="Defined FLERT Context variable")
def run_experiment(data_folder, task_name, model_name, run_id, use_context):
    # Set seed for reproducibility
    set_seed(int(run_id))

    if use_context == 0:
        use_context = False

    print("FLERT Context:", use_context)

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

    # Corpus configuration
    tag_type = "ner"
    tag_dictionary = corpus.make_tag_dictionary(tag_type=tag_type)
    print(tag_dictionary.idx2item)

    # Embeddings
    embedding_types: List[TokenEmbeddings] = [
        TransformerWordEmbeddings(model=model_name, layers="all", layer_mean=True, use_context=use_context)
    ]

    embeddings: StackedEmbeddings = StackedEmbeddings(embeddings=embedding_types)

    tagger: SequenceTagger = SequenceTagger(
        hidden_size=256,
        embeddings=embeddings,
        tag_dictionary=tag_dictionary,
        tag_type=tag_type,
        use_crf=True,
    )

    # Trainer
    trainer: ModelTrainer = ModelTrainer(tagger, corpus)

    trainer.train(
        f"resources/taggers/pos-{task_name}-{model_name}-context{use_context}-{run_id}",
        learning_rate=0.1,
        mini_batch_size=16,
        max_epochs=200,
        shuffle=True,
    )

if __name__ == "__main__":
    run_experiment()  # pylint: disable=no-value-for-parameter
