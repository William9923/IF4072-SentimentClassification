import os
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score
pd.options.mode.chained_assignment = None
from src.loader import DataLoader, ILoader
from src.preprocessor import TextPreprocessor, IPreprocessor
from src.feature_extractor import TFIDFFeatureExtractor, IBoWFeatureExtractor, IW2VFeatureExtractor
from src.classifier import LGBMClf, IClassifier
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
    loader:ILoader = DataLoader(target=target, sample_size=100, sampling=False, train_file_path=train_target_path, test_file_path=test_target_path)
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

    length = 1000
    vectorizer:IBoWFeatureExtractor = TFIDFFeatureExtractor(max_features=length, ngram_range=(1,1))
    
    if isinstance(vectorizer, IBoWFeatureExtractor):
        print("BoW")
        train_tokenized = vectorizer.fit_transform(X_train['cleaned_review'])
        val_tokenized = vectorizer.transform(X_val['cleaned_review'])
        test_tokenized = vectorizer.transform(X_test['cleaned_review'])

    if isinstance(vectorizer, IW2VFeatureExtractor):
        print("W2V")
        vectorizer.train(X_train['cleaned_review'].values)
        train_tokenized = vectorizer.tokenize(X_train['cleaned_review'], True)
        val_tokenized = vectorizer.tokenize(X_val['cleaned_review'], True)
        test_tokenized = vectorizer.tokenize(X_test['cleaned_review'], True)

    clf:IClassifier = LGBMClf(n_estimators=1000,learning_rate=0.1,max_depth=20)
    clf.train(X=train_tokenized, y=y_train, X_test=val_tokenized, y_test=y_val, early_stopping_rounds=10)
    pred = clf.predict(test_tokenized)
    print(f"Accuracy Score : {accuracy_score(y_test, pred)}")
    print(pred[:10].reshape(-1))
    print(loader.reverse_labels(pred[:10].reshape(-1)))
