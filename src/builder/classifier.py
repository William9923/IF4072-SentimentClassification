from tensorflow.keras.losses import BinaryCrossentropy
from tensorflow.keras.optimizers import Adam
from src.utility.config import Config
from src.classifier import IClassifier, FineTuneBertClf, LSTMClf, LGBMClf

# --- [Global Variable] ---
lstmClf: IClassifier = None
lgbmClf: IClassifier = None
bertClf: IClassifier = None


def build_lstm(config: Config) -> IClassifier:
    global lstmClf
    if lstmClf is not None:
        return lstmClf

    params = {
        "batch_size": config.batch_size,
        "length": config.max_vocab_size,
        "epochs": config.epochs,
        "embedding_matrix": config.embedding_matrix,
        "embedding_matrix_shape": (config.max_vocab_size, config.embedding_dimension),
        "loss": BinaryCrossentropy(),
        "optimizer": Adam(learning_rate=config.learning_rate_dl),
        "metrics": config.metrics,
    }
    clf = LSTMClf(**params)
    lstmClf = clf

    return clf


def build_lgbm(config: Config) -> IClassifier:
    global lgbmClf
    if lgbmClf is not None:
        return lgbmClf

    params = {
        "n_estimators": config.n_estimators,
        "learning_rate": config.learning_rate_sl,
        "max_depth": config.max_depth,
        "early_stopping_rounds": config.early_stopping_round
    }
    clf = LGBMClf(**params)
    lgbmClf = clf

    return clf


def build_bert(config: Config) -> IClassifier:
    global bertClf
    if bertClf is not None:
        return bertClf

    params = {
        "batch_size": config.batch_size,
        "length": config.pretrained_embedding_dimension,
        "epochs": config.epochs,
        "model_name": config.pretrained_model_name,
        "loss": BinaryCrossentropy(),
        "optimizer": Adam(learning_rate=config.learning_rate_dl),
        "metrics": config.metrics,
    }
    clf = FineTuneBertClf(**params)
    bertClf = clf

    return clf
