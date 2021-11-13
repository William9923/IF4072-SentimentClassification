#!/usr/bin/env bash

python main.py --name exp-p0-1.1 \
  --fe count \
  --clf nb \
  --no-sampling \
  --sample_size 100 \
  --target sentiment \
  --learning_rate 0.1 \
  --n_estimators 500 \
  --early_stopping_round 15 \
  --max_depth 24