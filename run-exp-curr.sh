#!/usr/bin/env bash

echo "📝 Running exp P1 3 : RoBERTa"
python main.py --name exp-p1-2.1 \
  --fe roberta \
  --clf roberta \
  --no-sampling \
  --sample_size 100 \
  --target sentiment \
  --model_name_or_path roberta-base \
  --learning_rate 4e-5 \
  --epochs 2 \
  --batch_size 12
  --max_length 30

echo "📝 Running exp P1 2.1 : BERT 1"
python main.py --name exp-p1-2.1 \
  --fe bert \
  --clf bert \
  --no-sampling \
  --sample_size 100 \
  --target sentiment \
  --model_name_or_path distilbert-base-uncased \
  --learning_rate 2.5e-5 \
  --epochs 2 \
  --batch_size 12
  --max_length 30

echo "📝 Running exp P1 2.2 : Bert 2"
python main.py --name exp-p1-2.2 \
  --fe bert \
  --clf bert \
  --no-sampling \
  --sample_size 100 \
  --target sentiment \
  --model_name_or_path distilbert-base-uncased \
  --learning_rate 3.5e-5 \
  --epochs 2 \
  --batch_size 12 \
  --max_length 30

echo "✅ Done running all experiment scenario ...♥️ "