# ========================================================
# Scripts for initialize sentiment classification pipeline
# ========================================================


import argparse
import os
import logging
import tensorflow as tf
tf.get_logger().setLevel(logging.ERROR)
import pandas as pd 
pd.options.mode.chained_assignment = None

from src.utility.constant import (
    LSTM_CLF_OPTION,
    TFIDF_FE_OPTION,
    COUNT_FE_OPTION,
    FASTTEXT_FE_OPTION,
    BERT_FE_OPTION,
    LGBM_CLF_OPTION,
    BERT_CLF_OPTION,
    TARGET,
    PRETRAINED_BERT_MODEL_NAME,
)

from src.utility.config import Config, Option
from pipeline import SentimentAnalyzer
from typing import Tuple


if __name__ == "__main__":

    parser = argparse.ArgumentParser()

    # --- [Option Settings] ---
    parser.add_argument(
        "--name", type=str, required=True, help=f"To describe the current experiment!"
    )
    parser.add_argument(
        "--fe",
        default=TFIDF_FE_OPTION,
        type=str,
        help=f"The name of feature extractor (word representation) for sentiment classification pipeline : {[TFIDF_FE_OPTION, COUNT_FE_OPTION, FASTTEXT_FE_OPTION, BERT_FE_OPTION]}.",
    )
    parser.add_argument(
        "--clf",
        default=LGBM_CLF_OPTION,
        type=str,
        help=f"The name of classifier for sentiment classification pipeline : {[LGBM_CLF_OPTION, LSTM_CLF_OPTION, BERT_CLF_OPTION]}.",
    )

    # --- [Config Settings] ---
    # Loader Settings
    parser.add_argument("--sampling", dest="sampling", action="store_true")
    parser.add_argument("--no-sampling", dest="sampling", action="store_false")
    parser.set_defaults(sampling=True)

    parser.add_argument("--sample_size", default=100, type=int)
    parser.add_argument("--target", default=TARGET, type=str)

    # Embedding Settings
    # parser.add_argument("--max_vocab_size", default=1000, type=int)
    # parser.add_argument("--embedding_dimension", default=512, type=int)
    # parser.add_argument("--min_count", default=1, type=int)
    # parser.add_argument("--window", default=5, type=int)
    # parser.add_argument("--num_words", default=1000, type=int)
    # parser.add_argument("--sg", default=1, type=int)

    parser.add_argument(
        "--max_length", type=int, default=30, help=f"To describe the max length for pre-trained tokenizer!"
    )

    # Training Settings
    parser.add_argument("--n_estimators", default=500, type=int)
    parser.add_argument("--early_stopping_round", default=15, type=int)
    parser.add_argument("--max_depth", default=24, type=int)


    parser.add_argument("--learning_rate", default=3e-4, type=float)
    parser.add_argument("--epochs", default=10, type=int)
    parser.add_argument(
        "--model_name_or_path", default=PRETRAINED_BERT_MODEL_NAME, type=str
    )

    parser.add_argument("--batch_size", default=32, type=int)

    args = parser.parse_args()

    option = Option(args.fe, args.clf)
    option.validate()

    if not os.path.exists("./bin"):
        os.mkdir("./bin")

    exp_dir = f"./bin/{args.name}"
    if not os.path.exists(exp_dir):
        os.mkdir(exp_dir)

    # --- [Setup config] ---
    config = Config(args.name)
    config.sampling = args.sampling
    config.sample_size = args.sample_size
    config.target = args.target

    # config.max_vocab_size = args.max_vocab_size
    # config.embedding_dimension = args.embedding_dimension
    # config.min_count = args.min_count
    # config.window = args.window
    # config.num_words = args.num_words
    # config.sg = args.sg
    config.max_length = args.max_length

    config.n_estimators = args.n_estimators
    config.early_stopping_round = args.early_stopping_round
    config.max_depth = args.max_depth

    if args.clf == LGBM_CLF_OPTION:
        config.learning_rate_sl = args.learning_rate
    else:
        config.learning_rate_dl = args.learning_rate

    config.epochs = args.epochs
    if args.model_name_or_path != "":
        config.pretrained_model_name = args.model_name_or_path
    config.batch_size = args.batch_size
    sentiment_analyzer = SentimentAnalyzer(config, option)
    sentiment_analyzer.build()

    print(f"Running Experiment : {config.experiment_name}")
    result = sentiment_analyzer.train()
    print("Result")
    print("Precision : ", result.get("precision"))
    print("Recall : ", result.get("recall"))
    print("F1 : ", result.get("f1"))
    print("Acc : ", result.get("accuracy"))

    sentiment_analyzer.save()
