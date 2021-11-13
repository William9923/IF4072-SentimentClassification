#!/usr/bin/env bash

python main.py --name exp-p1-1.3 \
  --fe fasttext \
  --clf lstm \
  --no-sampling \
  --sample_size 100 \
  --target sentiment \
  --learning_rate 1e-4 \
  --epochs 10 \
  --batch_size 12