import pandas as pd 

if __name__ == "__main__":
    train = pd.read_csv("./data/train.csv")
    test = pd.read_csv("./data/test.csv")
    used_cols = ['review', 'sentiment']

    print('training:')
    print(train.shape)
    print(train.columns)
    print("test:")
    print(test.shape)

    print(train['sentiment'].value_counts())
    encoder = {
            "neutral" : 0,
            "positive": 1,
            "negative": 2,
        }
    train['sentiment'] = train['sentiment'].map(encoder)
    print(train['sentiment'].value_counts())
    reverse_encoder = {}
    for key, val in encoder.items():
        reverse_encoder[val] = key
    train['sentiment'] = train['sentiment'].map(reverse_encoder)
    print(train['sentiment'].value_counts())
    # train[used_cols].to_csv("./data/train.csv", index=False)
    # test[used_cols].to_csv("./data/test.csv", index=False)