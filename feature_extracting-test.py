import os
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
pd.options.mode.chained_assignment = None
from src.loader import DataLoader, ILoader
from src.preprocessor import TextPreprocessor, IPreprocessor
from src.feature_extractor import CountFeatureExtractor, TFIDFFeatureExtractor, FastTextFeatureExtractor, BERTFeatureExtractor, IBoWFeatureExtractor, IW2VFeatureExtractor

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
    # vectorizer = FastTextFeatureExtractor(embedding_dimension=100, num_words=1000, min_count=1, window=5, sg=1)
    vectorizer = BERTFeatureExtractor("distilbert-base-uncased")
    if isinstance(vectorizer, IBoWFeatureExtractor):
        print("BoW")
    if isinstance(vectorizer, IW2VFeatureExtractor):
        print("W2V")
        vectorizer.train(X_train['cleaned_review'].values)
        print(vectorizer.get_embedding_matrix()[0:10])

