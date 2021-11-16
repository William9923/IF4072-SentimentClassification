import pandas as pd 
from nltk.tokenize import word_tokenize

if __name__ == "__main__":
    train = pd.read_csv("data/train.csv")
    test = pd.read_csv("data/test.csv")
    dataset = pd.concat([train, test])
    print(dataset.head())
    print(f"Total Data : {len(dataset)}")
    print(f"Training Data : {len(train)}")
    print(f"Test Data : {len(test)}")
    length = lambda x: len(word_tokenize(str(x)))
    print(f"Length Min: {dataset['review'].map(length).min()}")
    print(f"Length Max: {dataset['review'].map(length).max()}")
    print(f"Length Avg: {dataset['review'].map(length).mean()}")
    print(f"Length Median: {dataset['review'].map(length).median()}")
    