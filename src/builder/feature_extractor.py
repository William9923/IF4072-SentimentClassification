from src.utility.config import Config
from src.feature_extractor import (
    IBoWFeatureExtractor,
    IW2VFeatureExtractor,
    TFIDFFeatureExtractor,
    CountFeatureExtractor,
    FastTextFeatureExtractor,
    BERTFeatureExtractor,
)

# --- [Global Variable] ---
tfidfFE: IBoWFeatureExtractor = None
countFE: IBoWFeatureExtractor = None
fasttextFE: IW2VFeatureExtractor = None
bertFE: IW2VFeatureExtractor = None


def build_tfidf_fe(config: Config) -> IBoWFeatureExtractor:
    global tfidfFE
    if tfidfFE is not None:
        return tfidfFE

    params = {
        "max_features" : config.max_vocab_size,
        "ngram_range": (1,1)
    }
    fe = TFIDFFeatureExtractor(**params)
    tfidfFE = fe

    return fe


def build_count_fe(config: Config) -> IBoWFeatureExtractor:
    global countFE
    if countFE is not None:
        return countFE

    params = {
        "max_features" : config.max_vocab_size,
        "ngram_range": (1,1)
    }
    fe = CountFeatureExtractor(**params)
    countFE = fe

    return fe


def build_fasttext_fe(config: Config) -> IW2VFeatureExtractor:
    global fasttextFE
    if fasttextFE is not None:
        return fasttextFE

    params = {
        "embedding_dimension" : config.embedding_dimension,
        "num_words": config.num_words,
        "min_count": config.min_count,
        "window": config.window,
        "sg": config.sg,
    }
    fe = FastTextFeatureExtractor(**params)
    fasttextFE = fe

    return fe


def build_bert_fe(config: Config) -> IW2VFeatureExtractor:
    global bertFE
    if bertFE is not None:
        return bertFE

    params = {
        "pre_trained_name": config.pretrained_model_name,
        "max_length": config.max_length
    }
    fe = BERTFeatureExtractor(**params)
    bertFE = fe

    return fe

def build_roberta_fe(config: Config) -> IW2VFeatureExtractor:
    global bertFE
    if bertFE is not None:
        return bertFE

    params = {
        "pre_trained_name": config.pretrained_model_name,
        "max_length": config.max_length
    }
    fe = BERTFeatureExtractor(**params)
    bertFE = fe

    return fe
