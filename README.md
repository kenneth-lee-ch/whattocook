# What is cooking and what else should we put?

## Abstract
We explore the potential of creating new recipes via text data. Our goal has two folds. First, we aim to classify the cuisine based on ingredients. Second, we want to predict an ingredient that is missing from a given list of ingredients and a cuisine name. The first task can be formulated as a multi-class classification problem.  The second task can be viewed as a word prediction task. To convert the text into numerical signals, we can use a n-gram bag of words model with TF-IDF vectorizer, Countvectorizer. We will also consider building word embeddings with word2vec embedding, or other pre-trained embedding layers for training neural networks. 

For the first task, we can compare several well-known classification algorithms such as logistic regression, naive bayes, support vector machine, discriminant analysis. The evaluation methods will include F1-score, recall and precision. We will also approach the second task with recurrent neural networks, LSTM model, and some state-of-the-art neural network architectures such as Roberta. The second task will be evaluated by perplexity, which is the inverse probability of the test set, normalized by the number of words. 

## Data Source:
https://www.kaggle.com/c/whats-cooking/data
