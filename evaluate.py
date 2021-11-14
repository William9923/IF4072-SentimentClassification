import numpy as np
import pandas as pd 
pd.options.mode.chained_assignment = None
from src.utility.config import Config, Option
from pipeline import SentimentAnalyzer

evaluate_exp_name = "exp-p0-1.2"
evaluate_fe_option = "tfidf"
evaluate_clf_option = "nb"
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
human_label_pred = pipeline.loader.reverse_labels(pred)
for sample, label in zip(samples_text, human_label_pred):
    print(f"{sample} | {label}")