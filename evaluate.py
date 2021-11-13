import numpy as np
from os import pipe
import pandas as pd 
pd.options.mode.chained_assignment = None
from src.utility.config import Config, Option
from pipeline import SentimentAnalyzer

if __name__ == "__main__":
    evaluate_exp_name = "lgbm-testing"
    evaluate_fe_option = "tfidf"
    evaluate_clf_option = "lgbm"
    config = Config(evaluate_exp_name)
    option = Option(evaluate_fe_option, evaluate_clf_option)
    pipeline = SentimentAnalyzer(config, option)
    pipeline.build()
    pipeline.load()

    samples = [
        "I want to eat you",
        "that movie was amazing, we would like to see it again",
        "such an amazing performance",
        "that were terrible movie right there"
    ]

    labels = [0, 1, 1, 0]
    res, pred = pipeline.evaluate(pd.Series(np.array(samples)), labels)

    for sample, label in zip(samples, pred):
        print(f"{sample} | {label}")
