# ========================================================
# Scripts for initialize sentiment classification pipeline
# ========================================================

# SentimenAnalyzer class definition...

# inject... disini pokoknya semuanya

# Need to define experiment scopes :
# Changing parameter:

# DataLoader:
# SentimentClfLoader : ...

# == P0 Scenario ==
# Preprocessor: 
# W2VPreprocessor
# VectorSpacePreprocessor

# == P1 Scenario ==

# Word Representation + Parameter:  
# VectorSpace : 
# FastText : 
# BERT / Glove (pick 1) 

# == P2 Scenario ==
# Model:
# Default1 (Basic only -> kea di model yoga aja)
# Default2 (LGBM)
# SVM -> Traditional Machine Learning
# LSTM + Attention
# CNN

# == P3 Scenario ==
# Finding Best Parameter for the selected model...