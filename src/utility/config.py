import os

from tensorflow.python.keras.losses import BinaryCrossentropy
from tensorflow.python.keras.optimizer_v2.adam import Adam
from src.utility.constant import (
    TARGET,
    LOWERCASE_COMPONENT,
    MASK_EMOJI_COMPONENT,
    MASK_URL_COMPONENT,
    NORMALIZATION_COMPONENT,
    REMOVE_HTML_TAG_COMPONENT,
    REMOVE_PUNCT_COMPONENT,
    PRETRAINED_BERT_EMBEDDING_DIM,
    PRETRAINED_BERT_MODEL_NAME,
)


class Config(object):
    def __init__(self, exp_name: str):

        self.experiment_name = exp_name
        self.train_test_split = (0.8, 0.2)

        # ----- [Initializing Loader] ------
        self.sampling = False
        self.sample_size = 100
        self.target = TARGET
        self.train_file_path = os.path.join("data", "train.csv")
        self.test_file_path = os.path.join("data", "test.csv")
        # ----- [End Param] -----

        # ----- [Initializing Preprocessor] -----
        self.preprocessor_component = [
            LOWERCASE_COMPONENT,
            MASK_EMOJI_COMPONENT,
            MASK_URL_COMPONENT,
            NORMALIZATION_COMPONENT,
            REMOVE_HTML_TAG_COMPONENT,
            REMOVE_PUNCT_COMPONENT,
        ]
        # ----- [End Param] -----

        # ----- [Initializing Feature Extractor] -----
        self.max_vocab_size = 1000

        self.embedding_dimension = 512
        self.embedding_matrix = None
        self.min_count = 1
        self.window = 5
        self.sg = 1
        self.num_words = 10000

        self.pretrained_embedding_dimension = PRETRAINED_BERT_EMBEDDING_DIM
        # ----- [End Param] -----

        # ----- [Initializing Shallow ML Clf] -----
        self.n_estimators = 500
        self.learning_rate_sl = 0.1
        self.num_leaves = 20
        self.max_depth = 24
        self.early_stopping_round = 15
        # ----- [End Param] -----

        # ----- [Initializing Deep ML Clf] -----
        self.unit = 128
        self.learning_rate_dl = 3e-4
        self.pretrained_model_name = PRETRAINED_BERT_MODEL_NAME

        self.batch_size = 32
        self.epochs = 10

        self.metrics = ["accuracy"]
        # ----- [End Param] -----
