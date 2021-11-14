from gensim.models import FastText
from gensim.models.keyedvectors import KeyedVectors
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from transformers import (
    AutoTokenizer,
    RobertaTokenizerFast,
    RobertaConfig,
)

from src.feature_extractor.interface import IW2VFeatureExtractor
from src.utility.embedding import create_embedding_matrix


class FastTextFeatureExtractor(IW2VFeatureExtractor):
    def __init__(self, embedding_dimension, num_words, min_count, window, sg):
        self.window = window
        self.sg = sg
        self.num_words = num_words
        self.embedding_dimension = embedding_dimension
        self.min_count = min_count
        self.tokenizer = Tokenizer(num_words=num_words)
        self.fitted = False

    def train(self, X):
        self.embedding = FastText(
            sentences=X,
            vector_size=self.embedding_dimension,
            min_count=self.min_count,
            window=self.window,
            sg=self.sg,
        )
        self.tokenizer.fit_on_texts(X.tolist())
        self.embedding_matrix = create_embedding_matrix(
            self.tokenizer.word_index,
            self.num_words,
            self.embedding.wv,
            self.embedding_dimension,
        )
        self.fitted = True

    def tokenize(self, X, _):
        x = self.tokenizer.texts_to_sequences(X)
        x = pad_sequences(x, maxlen=self.num_words, padding="pre", truncating="post")
        return x

    def get_embedding_matrix(self):
        return self.embedding_matrix

    def save(self, filename):
        formatted_filename = f"{filename}.kv"
        self.embedding.wv.save(formatted_filename)

    def load(self, filename):
        self.embedding = FastText(
            vector_size=self.embedding_dimension,
            min_count=self.min_count,
            window=self.window,
            sg=self.sg,
        )
        formatted_filename = f"{filename}.kv"
        self.embedding.wv = KeyedVectors.load(formatted_filename)
        self.fitted = True


class BERTFeatureExtractor(IW2VFeatureExtractor):
    def __init__(self, pre_trained_name, max_length):
        self.tokenizer = AutoTokenizer.from_pretrained(pre_trained_name)
        self.max_length = max_length

    def train(self, _):
        print("Pre-trained tokenizer don't need to be trained!")

    def tokenize(self, X, mask_attention=False):
        x = self.tokenizer(
            list(X),
            max_length=self.max_length,
            padding="max_length",
            truncation=True,
            return_tensors="tf",
        )
        if mask_attention:
            x.data
        return x["input_ids"]

    def get_embedding_matrix(self):
        return self.embedding_matrix

    def save(self, _):
        print("Pre-trained tokenizer don't need to be saved!")

    def load(self, _):
        print("Pre-trained tokenizer don't need to be loaded!")


class RobertaFeatureExtractor(IW2VFeatureExtractor):
    def __init__(self, pre_trained_name, max_length):
        self.config = RobertaConfig.from_pretrained(pre_trained_name)
        self.tokenizer = RobertaTokenizerFast.from_pretrained(
            pretrained_model_name_or_path=pre_trained_name, config=self.config
        )
        self.max_length = max_length

    def train(self, _):
        print("Pre-trained tokenizer don't need to be trained!")

    def tokenize(self, X, mask_attention=False):
        x = self.tokenizer(
            X.tolist(),
            max_length=self.max_length,
            padding="max_length",
            truncation=True,
            return_tensors="tf",
        )
        if mask_attention:
            x.data
        return x["input_ids"]

    def get_embedding_matrix(self):
        pass

    def save(self, _):
        print("Pre-trained tokenizer don't need to be saved!")

    def load(self, _):
        print("Pre-trained tokenizer don't need to be loaded!")
