#!/usr/bin/env bash

echo "📝 Running exp P1 1.1 : LSTM | learning_rate 1.5e-5"
python main.py --name exp-p1-1.1 \
  --fe fasttext \
  --clf lstm \
  --no-sampling \
  --sample_size 100 \
  --target sentiment \
  --learning_rate 1.5e-5 \
  --epochs 10 \
  --batch_size 12

rm bin/exp-p1-1.1/extractor.kv.vectors_ngrams.npy

echo "📝 Running exp P1 1.2 : LSTM | learning_rate 5e-5"
python main.py --name exp-p1-1.2 \
  --fe fasttext \
  --clf lstm \
  --no-sampling \
  --sample_size 100 \
  --target sentiment \
  --learning_rate 5e-5 \
  --epochs 10 \
  --batch_size 12

rm bin/exp-p1-1.2/extractor.kv.vectors_ngrams.npy

echo "✅ Done running all experiment scenario ...♥️ "