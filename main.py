# ========================================================
# Scripts for initialize sentiment classification pipeline
# ========================================================

# SentimenAnalyzer class definition...
# TODO:
# Define the sentiment analyzer class definition
# build...
# train...
# evaluate...
# predict...
# show_example...

# inject... disini pokoknya semuanya

# Need to define experiment scopes :
# Changing parameter:

# DataLoader:
# SentimentClfLoader : ...

# == Init ==
# Preprocessor: 
# W2VPreprocessor
# VectorSpacePreprocessor

# == P1 Scenario (Jumat) -> Pick which word representation is the best (1 for default shallow ml, 1 for default deep learning) ==
# Word Representation + Parameter:  
# VectorSpace (for shallow ml only)
# FastText 
# BERT 

# == P2 Scenario (Jumat) -> Pick which classifier is the best from respective model (1 for shallow ml, 1 for deep learning model) ==
# Model:
# Default1 (vanilla)
# Default2 (LGBM)
# SVM -> Traditional Machine Learning
# LSTM + Attention
# Bert

# == P3 Scenario (Sabtu) ==
# Finding Best Parameter for the selected model... (1 for shallow ml, 1 for deep learning) -> gogo