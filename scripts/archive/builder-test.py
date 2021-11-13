import pandas as pd
from sklearn.metrics import accuracy_score
pd.options.mode.chained_assignment = None

from src.builder import build_data_loader, build_lgbm, build_tfidf_fe, build_text_prep
from src.utility.config import Config
from src.loader import ILoader
from src.preprocessor import IPreprocessor
from src.feature_extractor import IBoWFeatureExtractor, IW2VFeatureExtractor
from src.classifier import IClassifier


if __name__ == '__main__':
    config = Config("testing-builder")

    loader:ILoader = build_data_loader(config=config)
    loader.load()

    X_train, y_train = loader.get_train_data()
    X_val, y_val = loader.get_val_data()
    X_test, y_test = loader.get_test_data()

    preprocessor:IPreprocessor = build_text_prep(config=config)
    X_train['cleaned_review'] = preprocessor.preprocess(X_train['review'])
    X_val['cleaned_review'] = preprocessor.preprocess(X_val['review'])
    X_test['cleaned_review'] = preprocessor.preprocess(X_test['review'])

    extractor:IBoWFeatureExtractor = build_tfidf_fe(config=config)
    if isinstance(extractor, IBoWFeatureExtractor):
        train_tokenized = extractor.fit_transform(X_train['cleaned_review'])
        val_tokenized = extractor.transform(X_val['cleaned_review'])
        test_tokenized = extractor.transform(X_test['cleaned_review'])

    if isinstance(extractor, IW2VFeatureExtractor):
        extractor.train(X_train['cleaned_review'].values)
        train_tokenized = extractor.tokenize(X_train['cleaned_review'], True)
        val_tokenized = extractor.tokenize(X_val['cleaned_review'], True)
        test_tokenized = extractor.tokenize(X_test['cleaned_review'], True)
    
    clf:IClassifier = build_lgbm(config=config)
    clf.train(X=train_tokenized, y=y_train, X_test=val_tokenized, y_test=y_val)
    pred = clf.predict(test_tokenized)
    print(f"Accuracy Score : {accuracy_score(y_test, pred)}")
    print(pred[:10].reshape(-1))
    print(loader.reverse_labels(pred[:10].reshape(-1)))
