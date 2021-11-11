import joblib
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer

from src.feature_extractor.interface import IBoWFeatureExtractor

class BoWFeatureExtractor(IBoWFeatureExtractor):
    def __init__(self, embedding_dimension, ngram_range):
        self.max_features = embedding_dimension
        self.ngram_range = ngram_range
        self.fitted = False

    def fit(self, X):
        self.vectorizer.fit(X)
        self.fitted = True

    def transform(self, X):
        assert self.fitted
        return self.vectorizer.transform(X).toarray()

    def fit_transform(self, X):
        self.fitted = True
        return self.vectorizer.fit_transform(X).toarray()

    def save(self, filename):
        joblib.dump(self.vectorizer, filename)

    def load(self, filename):
        self.vectorizer = joblib.load(filename)

class CountFeatureExtractor(BoWFeatureExtractor):
    def __init__(self, max_features, ngram_range):
        super().__init__(max_features, ngram_range)
        self.vectorizer = CountVectorizer(max_features=max_features, ngram_range=ngram_range)

class TFIDFFeatureExtractor(BoWFeatureExtractor):
    def __init__(self, max_features, ngram_range):
        super().__init__(max_features, ngram_range)
        self.vectorizer = TfidfVectorizer(max_features=max_features, ngram_range=ngram_range)