#!/usr/bin/env bash

python main.py --name testing-lstm \
  --fe fasttext \
  --clf lstm \
  --sampling False \
  --sample_size 100 \
  --target sentiment \
  --epochs 10 \
  --batch_size 32 \