#!/usr/bin/env bash

python main.py --name testing-lgbm \
  --fe tfidf \
  --clf lgbm \
  --sampling \
  --sample_size 100 \
  --target sentiment \
  --learning_rate 0.1 \
  --n_estimators 500 \
  --early_stopping_round 15 \
  --max_depth 24