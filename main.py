# ========================================================
# Scripts for initialize sentiment classification pipeline
# ========================================================

# SentimenAnalyzer class definition...

# inject... disini pokoknya semuanya

# Need to define experiment scopes :
# Changing parameter:

# DataLoader:
# SentimentClfLoader : ...

# == P0 ==
# Preprocessor: 
# W2VPreprocessor
# VectorSpacePreprocessor

# == P1 Scenario -> Pick which word representation is the best (1 for default shallow ml, 1 for default deep learning) ==
# Word Representation + Parameter:  
# VectorSpace : 
# FastText : 
# BERT / Glove (pick 1) 

# == P2 Scenario -> Pick which classifier is the best from respective model (1 for shallow ml, 1 for deep learning model) ==
# Model:
# Default1 (Basic only -> kea di model yoga aja)
# Default2 (LGBM)
# SVM -> Traditional Machine Learning
# LSTM + Attention
# CNN

# == P3 Scenario ==
# Finding Best Parameter for the selected model... (1 for shallow ml, 1 for deep learning)