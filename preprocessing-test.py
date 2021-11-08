import os
import pandas as pd 
pd.options.mode.chained_assignment = None
from src.loader import DataLoader
from src.preprocessor import TextPreprocessor

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
    loader = DataLoader(target=target, sample_size=100, sampling=True, train_file_path=train_target_path, test_file_path=test_target_path)
    loader.load()

    X_train, y_train = loader.get_train_data()

    component = [
        LOWERCASE_COMPONENT,
        MASK_EMOJI_COMPONENT, 
        MASK_URL_COMPONENT, 
        NORMALIZATION_COMPONENT, 
        REMOVE_HTML_TAG_COMPONENT, 
        REMOVE_PUNCT_COMPONENT,
        EMOJI_MASK,
    ]
    preprocessor = TextPreprocessor(component=component)
    X_train['cleaned_review'] = preprocessor.preprocess(X_train['review'])

    print(X_train[['review', 'cleaned_review']].sample(10))