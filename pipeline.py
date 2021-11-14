import os
import json
import pandas as pd
from src.classifier.impl.baseline import LSTMClf

pd.options.mode.chained_assignment = None
from gensim.utils import simple_preprocess

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
    NB_CLF_OPTION,
)

from src.utility.config import Option, Config

from src.builder import (
    build_lstm,
    build_lgbm,
    build_bert,
    build_nb,
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
            NB_CLF_OPTION: build_nb,
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

    def train(self):
        assert self.compiled
        self.loader.load()

        X_train, y_train = self.loader.get_train_data()
        X_val, y_val = self.loader.get_val_data()
        X_test, y_test = self.loader.get_test_data()

        # X_train['cleaned_review'] = X_train['review'].apply(simple_preprocess)
        # print(X_val['review'])
        # X_val['cleaned_review'] = X_val['review'].apply(simple_preprocess)
        # X_test['cleaned_review'] = X_test['review'].apply(simple_preprocess)

        X_train["cleaned_review"] = self.preprocessor.preprocess(X_train["review"])
        X_val["cleaned_review"] = self.preprocessor.preprocess(X_val["review"])
        X_test["cleaned_review"] = self.preprocessor.preprocess(X_test["review"])

        if isinstance(self.extractor, IBoWFeatureExtractor):
            train_tokenized = self.extractor.fit_transform(X_train["cleaned_review"])
            val_tokenized = self.extractor.transform(X_val["cleaned_review"])
            test_tokenized = self.extractor.transform(X_test["cleaned_review"])

        if isinstance(self.extractor, IW2VFeatureExtractor):
            self.extractor.train(X_train["cleaned_review"].values)

            if isinstance(self.classifier, LSTMClf):
                self.classifier.set_embedding_matrix(
                    self.extractor.get_embedding_matrix()
                )

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
        self.result = {
            "precision": precision_score(y_test, pred, average='micro'),
            "recall": recall_score(y_test, pred,average='micro'),
            "f1": f1_score(y_test, pred, average='micro'),
            "accuracy": accuracy_score(y_test, pred),
        }
        return self.result

    def save(self):
        filename_prefix = os.path.join("bin", self.config.experiment_name)
        if isinstance(self.extractor, IBoWFeatureExtractor) or isinstance(
            self.extractor, IW2VFeatureExtractor
        ):
            self.extractor.save(os.path.join(filename_prefix, "extractor"))
        self.classifier.save(os.path.join(filename_prefix, "model"))

        log(
            filename_prefix,
            str(self.option).upper(),
            self.__parse_config(self.option),
        )
        log(
            filename_prefix,
            str(self.config).upper(),
            self.__parse_config(self.config),
        )
        log(filename_prefix, "EVALUATION", self.result)

    def load(self):
        assert self.compiled
        filename_prefix = os.path.join("bin", self.config.experiment_name)
        self.extractor.load(os.path.join(filename_prefix, "extractor"))
        self.classifier.load(os.path.join(filename_prefix, "model"))
        self.trained = True

    def evaluate(self, batch, labels):
        assert self.trained
        prep_batch = self.preprocessor.preprocess(batch)

        if isinstance(self.extractor, IBoWFeatureExtractor):
            prep_tokenized = self.extractor.transform(prep_batch)

        if isinstance(self.extractor, IW2VFeatureExtractor):
            get_attention_mask = self.option.clf_option == BERT_CLF_OPTION
            prep_tokenized = self.extractor.tokenize(prep_batch, get_attention_mask)

        pred = self.classifier.predict(prep_tokenized)
        result = {
            "precision": precision_score(labels, pred, average='micro'),
            "recall": recall_score(labels, pred,average='micro'),
            "f1": f1_score(labels, pred, average='micro'),
            "accuracy": accuracy_score(labels, pred),
        }

        return result, pred

    def __parse_config(self, obj):
        keys = [
            a
            for a in dir(obj)
            if not a.startswith("__") and not callable(getattr(obj, a))
        ]
        dicts = {}
        for key in keys:
            dicts[key] = getattr(obj, key)
        return dicts


def log(path, topic, dicts):
    dicts["topic"] = topic
    with open(os.path.join(path, topic + ".json"), "w", encoding="utf-8") as f:
        json.dump(dicts, f, ensure_ascii=False, indent=4)
