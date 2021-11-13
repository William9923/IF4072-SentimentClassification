#!/usr/bin/env bash

python main.py --name lgbm-testing \
  --fe tfidf \
  --clf lgbm \
  --no-sampling \
  --sample_size 100 \
  --target sentiment \
  --learning_rate 0.1 \
  --n_estimators 500 \
  --early_stopping_round 15 \
  --max_depth 24