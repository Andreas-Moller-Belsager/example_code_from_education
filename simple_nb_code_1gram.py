import os
import re
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.preprocessing import LabelEncoder
from sklearn.naive_bayes import GaussianNB, MultinomialNB
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline


X_dev = np.load('../models/100feats_1grams_dev_X.npy', allow_pickle=True)[()]
y_dev = np.load('../models/100feats_1grams_dev_y.npy', allow_pickle=True)
X_test = np.load('../models/100feats_1grams_test_X.npy', allow_pickle=True)[()]
y_test = np.load('../models/100feats_1grams_test_y.npy', allow_pickle=True)
X_train = np.load('../models/100feats_1grams_train_X.npy', allow_pickle=True)[()]
y_train = np.load('../models/100feats_1grams_train_y.npy', allow_pickle=True)

lr = MultinomialNB()
fit_lr = lr.fit(X_train, y_train)
y_pred_lr_dev = fit_lr.predict(X_dev)
y_pred_lr_test = fit_lr.predict(X_test)

print('MultinomialNB F1 (1 grams, dev, macro): ', f1_score(y_dev, y_pred_lr_dev, average='macro')) 
print('MultinomialNB F1 (1 grams, train, macro): ', f1_score(y_test, y_pred_lr_test, average='macro')) 
print('MultinomialNB Accuracy (1 grams, dev): ', accuracy_score(y_dev, y_pred_lr_dev)) 
print('MultinomialNB Accuracy (1 grams, train): ', accuracy_score(y_test, y_pred_lr_test)) 