# --- [Constant running process] ---
SEED = 123
CONFIG_CLS = "config"
OPTION_CLS = "option"

# --- [Preprocessor Component] ---
LOWERCASE_COMPONENT = "lower"
MASK_URL_COMPONENT = "mask.url"
REMOVE_HTML_TAG_COMPONENT = "remove.html.tag"
MASK_EMOJI_COMPONENT = "mask.emoji"
REMOVE_PUNCT_COMPONENT = "remove.punct"
NORMALIZATION_COMPONENT = "normalization"

EMOJI_MASK = "emoji"

# --- [Classification Related] ---
TARGET = "sentiment"
PRETRAINED_BERT_EMBEDDING_DIM = 512
PRETRAINED_BERT_MODEL_NAME = "distilbert-base-uncased"

# --- [Experiment Option Related] ---
COUNT_FE_OPTION = "count"
TFIDF_FE_OPTION = "tfidf"
FASTTEXT_FE_OPTION = "fasttext"
BERT_FE_OPTION = "bert"

NB_CLF_OPTION = "nb"
LGBM_CLF_OPTION = "lgbm"
LSTM_CLF_OPTION = "lstm"
BERT_CLF_OPTION = "bert"