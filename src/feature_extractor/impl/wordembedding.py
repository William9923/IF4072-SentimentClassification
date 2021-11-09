from gensim.models import FastText
from tensorflow.keras.preprocessing.text import Tokenizer
import transformers
from tensorflow.keras.preprocessing.sequence import pad_sequences

from src.feature_extractor.interface import IW2VFeatureExtractor
from src.utility.embedding import create_embedding_matrix

class FastTextFeatureExtractor(IW2VFeatureExtractor):
    def __init__(self, embedding_dimension, num_words, min_count, window, sg):
        self.window = window
        self.sg = sg 
        self.num_words = num_words
        self.embedding_dimension = embedding_dimension
        self.min_count = min_count
        self.tokenizer = Tokenizer()
        self.fitted = False

    def train(self, X):
        self.embedding = FastText(sentences=X, vector_size=self.embedding_dimension, min_count=self.min_count, window=self.window, sg=self.sg)
        self.tokenizer.fit_on_texts(X.tolist())
        self.embedding_matrix = create_embedding_matrix(self.tokenizer.word_index, self.num_words, self.embedding.wv, self.embedding_dimension)
        self.fitted = True

    def tokenize(self, X):
        x = self.tokenizer.texts_to_sequences(X)
        x = pad_sequences(x, maxlen=self.max_length, padding="pre", truncating="post")
        return x

    def get_embedding_matrix(self):
        return self.embedding_matrix

    def save(self, filename):
        self.embedding.save(filename)

    def load(self, filename):
        self.embedding = FastText.load(filename)
        self.fitted = True

class BERTFeatureExtractor(IW2VFeatureExtractor):
    def __init__(self, pre_trained_name):
        self.embedding = transformers.TFDistilBertModel.from_pretrained(pre_trained_name)
        self.tokenizer = transformers.AutoTokenizer.from_pretrained(pre_trained_name)
        self.embedding_matrix = self.embedding.weights[0].numpy()

    def train(self, X):
        pass

    def tokenize(self, X):
        x = self.tokenizer(list(X), padding='max_length', truncation=True, return_tensors="tf")
        return x['input_ids']

    def get_embedding_matrix(self):
        return self.embedding_matrix

    def save(self, filename):
        pass

    def load(self, filename):
        pass

