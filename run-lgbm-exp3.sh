#!/usr/bin/env bash

python main.py --name exp-p0-2.3 \
  --fe tfidf \
  --clf lgbm \
  --no-sampling \
  --sample_size 100 \
  --target sentiment \
  --learning_rate 0.05 \
  --n_estimators 250 \
  --early_stopping_round 15 \
  --max_depth 24