#!/usr/bin/env bash

python main.py --name testing \
  --fe count \
  --clf lgbm \
  --sampling False \
  --sample_size 100 \
  --target sentiment \
  --max_vocab_size 1000 \
  --n_estimators 500 \
  --early_stopping_round 15 \
  --max_depth 24 