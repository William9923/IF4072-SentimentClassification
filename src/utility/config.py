import os

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
    COUNT_FE_OPTION,
    TFIDF_FE_OPTION,
    FASTTEXT_FE_OPTION,
    BERT_FE_OPTION,
    ROBERTA_FE_OPTION,
    LGBM_CLF_OPTION,
    LSTM_CLF_OPTION,
    BERT_CLF_OPTION,
    NB_CLF_OPTION,
    ROBERTA_CLF_OPTION,
    CONFIG_CLS,
    OPTION_CLS,
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
        self.num_words = 1000

        self.pretrained_embedding_dimension = PRETRAINED_BERT_EMBEDDING_DIM
        self.max_length = 30
        # ----- [End Param] -----

        # ----- [Initializing Shallow ML Clf] -----
        self.n_estimators = 500
        self.learning_rate_sl = 0.1
        self.num_leaves = 20
        self.max_depth = 24
        self.early_stopping_round = 15
        # ----- [End Param] -----

        # ----- [Initializing Deep ML Clf] -----
        self.learning_rate_dl = 1.5e-5
        self.pretrained_model_name = PRETRAINED_BERT_MODEL_NAME

        self.batch_size = 32
        self.epochs = 10

        self.metrics = ["accuracy"]
        # ----- [End Param] -----

    def __str__(self):
        return CONFIG_CLS


class Option:
    def __init__(self, fe_option, clf_option):
        self.fe_option = fe_option
        self.clf_option = clf_option

    def __str__(self):
        return OPTION_CLS

    def validate(self):

        if self.fe_option not in [
            COUNT_FE_OPTION,
            TFIDF_FE_OPTION,
            FASTTEXT_FE_OPTION,
            BERT_CLF_OPTION,
        ]:
            raise Exception("Feature Extractor Option not exist!")

        if self.clf_option not in [
            NB_CLF_OPTION,
            LGBM_CLF_OPTION,
            LSTM_CLF_OPTION,
            BERT_CLF_OPTION,
            ROBERTA_CLF_OPTION,
        ]:
            raise Exception("Classifier Option not exist!")

        supported = False
        # --- [Shallow ML Validation] ---
        if self.fe_option in [COUNT_FE_OPTION, TFIDF_FE_OPTION] and self.clf_option in [
            LGBM_CLF_OPTION,
            NB_CLF_OPTION,
        ]:
            supported = True

        if (
            self.fe_option in [FASTTEXT_FE_OPTION, BERT_FE_OPTION]
            and self.clf_option == LSTM_CLF_OPTION
        ):
            supported = True

        if self.fe_option in [BERT_FE_OPTION] and self.clf_option == BERT_CLF_OPTION:
            supported = True

        if self.fe_option in [ROBERTA_FE_OPTION] and self.clf_option == ROBERTA_CLF_OPTION:
            supported = True

        if not supported:
            raise Exception("Experiment not supported with this pipeline!")
