import os
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score
from sklearn.feature_extraction.text import TfidfVectorizer
pd.options.mode.chained_assignment = None
from src.loader import DataLoader, ILoader
from src.preprocessor import TextPreprocessor, IPreprocessor
from src.feature_extractor import CountFeatureExtractor, TFIDFFeatureExtractor, FastTextFeatureExtractor, BERTFeatureExtractor, IBoWFeatureExtractor, IW2VFeatureExtractor
from src.classifier import LSTMClf
from src.utility.constant import (
    LOWERCASE_COMPONENT,
    MASK_EMOJI_COMPONENT, 
    MASK_URL_COMPONENT, 
    NORMALIZATION_COMPONENT, 
    REMOVE_HTML_TAG_COMPONENT, 
    REMOVE_PUNCT_COMPONENT,
    EMOJI_MASK,
)
if __name__ == "__main__":
    target = "sentiment"
    train_target_path = os.path.join("./data", "train.csv")
    test_target_path = os.path.join("./data", "test.csv")
    loader:ILoader = DataLoader(target=target, sample_size=100, sampling=True, train_file_path=train_target_path, test_file_path=test_target_path)
    loader.load()

    X_train, y_train = loader.get_train_data()
    X_val, y_val = loader.get_val_data()
    X_test, y_test = loader.get_test_data()

    component = [
        LOWERCASE_COMPONENT,
        MASK_EMOJI_COMPONENT, 
        MASK_URL_COMPONENT, 
        NORMALIZATION_COMPONENT, 
        REMOVE_HTML_TAG_COMPONENT, 
        REMOVE_PUNCT_COMPONENT,
        EMOJI_MASK,
    ]
    preprocessor:IPreprocessor = TextPreprocessor(component=component)
    X_train['cleaned_review'] = preprocessor.preprocess(X_train['review'])
    X_val['cleaned_review'] = preprocessor.preprocess(X_val['review'])
    X_test['cleaned_review'] = preprocessor.preprocess(X_test['review'])
    vectorizer = FastTextFeatureExtractor(embedding_dimension=100, num_words=100, min_count=1, window=5, sg=1)
    # vectorizer = BERTFeatureExtractor(1000, "distilbert-base-uncased")
    if isinstance(vectorizer, IBoWFeatureExtractor):
        print("BoW")
    if isinstance(vectorizer, IW2VFeatureExtractor):
        print("W2V")
        vectorizer.train(X_train['cleaned_review'].values)
        print(vectorizer.fitted)

    train_tokenized = vectorizer.tokenize(X_train['cleaned_review'])
    val_tokenized = vectorizer.tokenize(X_val['cleaned_review'])
    test_tokenized = vectorizer.tokenize(X_test['cleaned_review'])
    
    # save_path = os.path.join("bin", "bert-vectorizer.pkl")
    clf = LSTMClf(32, 100, embedding_matrix=vectorizer.get_embedding_matrix())
    clf.train(X=train_tokenized, y=y_train, X_test=val_tokenized, y_test=y_val, epochs=10)
    pred = clf.predict(test_tokenized)
    print(f"Accuracy Score : {accuracy_score(y_test, pred)}")
    print(pred[:10].reshape(-1))
    print(loader.reverse_labels(pred[:10].reshape(-1)))


