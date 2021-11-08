import pandas as pd 
import os 
from sklearn.model_selection import train_test_split

from src.loader import DataLoader


if __name__ == '__main__':

    target = "sentiment"
    train_target_path = os.path.join("./data", "train.csv")
    test_target_path = os.path.join("./data", "test.csv")
    loader = DataLoader(target=target, sample_size=100, sampling=False, train_file_path=train_target_path, test_file_path=test_target_path)
    loader.load()
    X_train, y_train = loader.get_train_data()
    X_val, y_val = loader.get_val_data()
    X_test, y_test = loader.get_test_data()

    print("Training data")
    print(X_train.shape)
    print(y_train.shape)

    print(X_train.head())
    print(y_train.head())

    print("Testing data")
    print(X_test.shape)
    print(y_test.shape)

    print(X_test.head())
    print(y_test.head())

    print("Validation data")
    print(X_val.shape)
    print(y_val.shape)

    print(X_val.head())
    print(y_val.head())

    print(loader.reverse_labels(y_val.values[:100]))

    # filename = "dataset.csv"
    # file_path = os.path.join("./data", filename)
    # target = "sentiment"
    # data = pd.read_csv(file_path)
    # train = data[:40000]
    # test = data[40000:]

    # print(train.shape)
    # print(test.shape)

    # train_target_path = os.path.join("./data", "train.csv")
    # train.to_csv(train_target_path, index=False)

    # test_target_path = os.path.join("./data", "test.csv")
    # test.to_csv(test_target_path, index=False)