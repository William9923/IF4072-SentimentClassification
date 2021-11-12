#!/usr/bin/env bash

python main.py --name testing-lstm \
  --fe fasttext \
  --clf lstm \
  --no-sampling \
  --sample_size 100 \
  --target sentiment \
  --learning_rate 3e-4 \
  --epochs 10 \
  --batch_size 32