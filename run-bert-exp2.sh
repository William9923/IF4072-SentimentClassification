#!/usr/bin/env bash

python main.py --name exp-p1-2.2 \
  --fe bert \
  --clf bert \
  --no-sampling \
  --sample_size 100 \
  --target sentiment \
  --model_name_or_path distilbert-base-uncased \
  --learning_rate 5e-5 \
  --epochs 2 \
  --batch_size 12