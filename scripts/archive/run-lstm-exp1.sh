#!/usr/bin/env bash

python main.py --name exp-p1-1.1 \
  --fe fasttext \
  --clf lstm \
  --no-sampling \
  --sample_size 100 \
  --target sentiment \
  --learning_rate 1.5e-5 \
  --epochs 10 \
  --batch_size 12