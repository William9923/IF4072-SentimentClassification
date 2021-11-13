#!/usr/bin/env bash

echo "📝 Running exp P0 1.1 : Naive Bayes | Count Vectorizer ..."
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

echo "📝 Running exp P0 1.2 : Naive Bayes | TFiDF Vectorizer ..."
python main.py --name exp-p0-1.2 \
  --fe tfidf \
  --clf nb \
  --no-sampling \
  --sample_size 100 \
  --target sentiment \
  --learning_rate 0.1 \
  --n_estimators 500 \
  --early_stopping_round 15 \
  --max_depth 24

echo "📝 Running exp P0 2.1 : LGBM | n_estimators 500, learning+rate 0.05"
python main.py --name exp-p0-2.1 \
  --fe tfidf \
  --clf lgbm \
  --no-sampling \
  --sample_size 100 \
  --target sentiment \
  --learning_rate 0.05 \
  --n_estimators 500 \
  --early_stopping_round 15 \
  --max_depth 24

echo "📝 Running exp P0 2.2 : LGBM | n_estimators 500, learning+rate 0.1"
  python main.py --name exp-p0-2.2 \
  --fe tfidf \
  --clf lgbm \
  --no-sampling \
  --sample_size 100 \
  --target sentiment \
  --learning_rate 0.1 \
  --n_estimators 500 \
  --early_stopping_round 15 \
  --max_depth 24

echo "📝 Running exp P0 2.3 : LGBM | n_estimators 250, learning+rate 0.05"
python main.py --name exp-p0-2.3 \
  --fe tfidf \
  --clf lgbm \
  --no-sampling \
  --sample_size 100 \
  --target sentiment \
  --learning_rate 0.05 \
  --n_estimators 250 \
  --early_stopping_round 15 \
  --max_depth 24

echo "📝 Running exp P0 2.4 : LGBM | n_estimators 250, learning+rate 0.1"
python main.py --name exp-p0-2.4 \
  --fe tfidf \
  --clf lgbm \
  --no-sampling \
  --sample_size 100 \
  --target sentiment \
  --learning_rate 0.1 \
  --n_estimators 250 \
  --early_stopping_round 15 \
  --max_depth 24

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

echo "📝 Running exp P1 2.1 : BERT | learning_rate 1.5e-5"
python main.py --name exp-p1-2.1 \
  --fe bert \
  --clf bert \
  --no-sampling \
  --sample_size 100 \
  --target sentiment \
  --model_name_or_path distilbert-base-uncased \
  --learning_rate 1.5e-5 \
  --epochs 2 \
  --batch_size 12

echo "📝 Running exp P1 2.2 : Bert | learning_rate 5e-5"
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

echo "✅ Done running all experiment scenario ...♥️ "