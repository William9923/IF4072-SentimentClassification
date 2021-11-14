import numpy as np
import pandas as pd 
pd.options.mode.chained_assignment = None
from src.utility.config import Config, Option
from pipeline import SentimentAnalyzer

evaluate_exp_name = "testing-lstm"
evaluate_fe_option = "fasttext"
evaluate_clf_option = "lstm"
config = Config(evaluate_exp_name)
option = Option(evaluate_fe_option, evaluate_clf_option)
pipeline = SentimentAnalyzer(config, option)

pipeline.build()
pipeline.load()

test = pd.read_csv("data/test.csv")
samples = test.head(20)
samples_text = samples['review']
labels = samples['sentiment'].map({'neutral': 0,'negative': 2, 'positive': 1})
res, pred = pipeline.evaluate(samples_text, labels)

for sample, label in zip(samples_text, pred):
    print(f"{sample} | {label}")