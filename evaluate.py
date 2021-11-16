import numpy as np
import pandas as pd 
pd.options.mode.chained_assignment = None
from src.utility.config import Config, Option
from pipeline import SentimentAnalyzer

evaluate_exp_name = "exp-p1-2.1"
evaluate_fe_option = "bert"
evaluate_clf_option = "bert"
config = Config(evaluate_exp_name)
option = Option(evaluate_fe_option, evaluate_clf_option)
pipeline = SentimentAnalyzer(config, option)

pipeline.build()
pipeline.load()

test = pd.read_csv("data/test.csv")
samples_text = test['review']
labels = test['sentiment'].map({'neutral': 0,'negative': 2, 'positive': 1})
res, pred = pipeline.evaluate(samples_text, labels)
human_label_pred = pipeline.loader.reverse_labels(pred)
result = pd.DataFrame({
    "text" : test['review'],
    "label": human_label_pred
})

result.to_csv("prediction.csv", index=False)