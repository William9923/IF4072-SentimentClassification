import pandas as pd 
pd.options.mode.chained_assignment = None

from src.classifier.interface import IClassifier
from src.loader.interface import ILoader
from src.feature_extractor.interface import IBoWFeatureExtractor, IW2VFeatureExtractor
from src.preprocessor.interface import IPreprocessor
from src.utility.constant import (
    COUNT_FE_OPTION,
    TFIDF_FE_OPTION,
    FASTTEXT_FE_OPTION,
    BERT_FE_OPTION,
    LGBM_CLF_OPTION,
    LSTM_CLF_OPTION,
    BERT_CLF_OPTION,
)

from src.utility.config import Option, Config

from src.builder import (
    build_lstm,
    build_lgbm,
    build_bert,
    build_bert_fe,
    build_fasttext_fe,
    build_count_fe,
    build_tfidf_fe,
    build_text_prep,
    build_data_loader,
)

from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score


class SentimentAnalyzer:
    def __init__(self, config: Config, option: Option):
        self.config = config
        self.option = option
        self.loader: ILoader = None
        self.preprocessor: IPreprocessor = None
        self.feature_extractor = None
        self.classifier: IClassifier = None
        self.fe_map = {
            COUNT_FE_OPTION: build_count_fe,
            TFIDF_FE_OPTION: build_tfidf_fe,
            FASTTEXT_FE_OPTION: build_fasttext_fe,
            BERT_FE_OPTION: build_bert_fe,
        }

        self.clf_map = {
            LGBM_CLF_OPTION: build_lgbm,
            LSTM_CLF_OPTION: build_lstm,
            BERT_CLF_OPTION: build_bert,
        }
        self.compiled = False
        self.trained = False

    def build(self):
        self.loader = build_data_loader(self.config)
        self.preprocessor = build_text_prep(self.config)
        self.extractor = self.fe_map[self.option.fe_option](self.config)
        self.classifier = self.clf_map[self.option.clf_option](self.config)
        self.compiled = True

    def run(self):
        assert self.compiled
        self.loader.load()

        X_train, y_train = self.loader.get_train_data()
        X_val, y_val = self.loader.get_val_data()
        X_test, y_test = self.loader.get_test_data()

        X_train["cleaned_review"] = self.preprocessor.preprocess(X_train["review"])
        X_val["cleaned_review"] = self.preprocessor.preprocess(X_val["review"])
        X_test["cleaned_review"] = self.preprocessor.preprocess(X_test["review"])

        if isinstance(self.extractor, IBoWFeatureExtractor):
            train_tokenized = self.extractor.fit_transform(X_train["cleaned_review"])
            val_tokenized = self.extractor.transform(X_val["cleaned_review"])
            test_tokenized = self.extractor.transform(X_test["cleaned_review"])

        if isinstance(self.extractor, IW2VFeatureExtractor):
            self.extractor.train(X_train["cleaned_review"].values)

            get_attention_mask = self.option.clf_option == BERT_CLF_OPTION
            train_tokenized = self.extractor.tokenize(
                X_train["cleaned_review"], get_attention_mask
            )
            val_tokenized = self.extractor.tokenize(
                X_val["cleaned_review"], get_attention_mask
            )
            test_tokenized = self.extractor.tokenize(
                X_test["cleaned_review"], get_attention_mask
            )

        self.classifier.train(
            X=train_tokenized, y=y_train, X_test=val_tokenized, y_test=y_val
        )

        self.trained = True
        pred = self.classifier.predict(test_tokenized)
        return {
            "precision": precision_score(y_test, pred),
            "recall": recall_score(y_test, pred),
            "f1": f1_score(y_test, pred),
            "accuracy": accuracy_score(y_test, pred),
        }

    def save(self):
        pass

    def load(self):
        pass

    def interactive_run(self, batch):
        assert self.compiled
        assert self.trained
