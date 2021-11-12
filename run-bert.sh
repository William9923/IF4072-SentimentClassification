#!/usr/bin/env bash

python main.py --name testing-bert \
  --fe bert \
  --clf bert \
  --no-sampling \
  --sample_size 100 \
  --target sentiment \
  --model_name_or_path distilbert-base-uncased \
  --learning_rate 3e-4 \
  --epochs 2 \
  --batch_size 12