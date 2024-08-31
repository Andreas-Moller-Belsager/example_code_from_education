import os
import re
import pickle
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


for i in range(10):
	thisnumber = i+1
	ngramrange = (thisnumber,thisnumber)

	def create_vectorizer(directory):
		targets = []
		featureset = set()
		
		for filename in os.listdir(directory):
			corpus = []
			with open(directory+filename, 'r', errors='ignore') as file:
				i=0
				for line in file:
					if not i== 100:
						corpus.append(line)
						i+=1
			
			vectorizer = CountVectorizer(ngram_range = ngramrange, analyzer = 'char', max_features=100, dtype=np.int32)
		
			vectorizer.fit_transform(corpus)
			feats = vectorizer.get_feature_names()
			#print(feats)
			#print(type(feats))
			for el in feats:
				featureset.add(el)
			
		return featureset
		
		
		


	def preprocess_text(directory, num_lines):
		full_corpus = []
		full_targets = []
		for filename in os.listdir(directory):
			#print(filename)
			file_elements = filename.split('.')
			result = file_elements[2]
			fullname = directory+filename
			if num_lines == 'full':
				with open(fullname, 'r', errors='ignore') as file:     #he said I could use [:1000] to subset it further, so it takes shorter time
					for line in file:
						full_corpus.append(line.strip())
						full_targets.append(result)
			else:
				with open(fullname, 'r', errors='ignore') as file:     #he said I could use [:1000] to subset it further, so it takes shorter time
					i=0
					for line in file:
						if not i==num_lines:
							full_corpus.append(line.strip())
							full_targets.append(result)
							i+=1

		
		return full_corpus, full_targets



	featureset = create_vectorizer('MIL-TALE/4/data_dir/train/')


	X_train, y_train = preprocess_text("MIL-TALE/4/data_dir/train/", 100)
	X_dev, y_dev = preprocess_text("MIL-TALE/4/data_dir/devtest/", 'full')
	X_test, y_test = preprocess_text("MIL-TALE/4/data_dir/test/", 'full')


	train_corpus = []
	train_targets = []
	for el1, el2 in zip(X_train, y_train):
		if (el2 in y_test) and (el2 in y_dev):
			train_corpus.append(el1)
			train_targets.append(el2)

	test_corpus = []
	test_targets = []
	for el1, el2 in zip(X_test, y_test):
		if el2 in train_targets:
			test_corpus.append(el1)
			test_targets.append(el2)
			

	dev_corpus = []
	dev_targets = []
	for el1, el2 in zip(X_dev, y_dev):
		if el2 in train_targets:
			dev_corpus.append(el1)
			dev_targets.append(el2)


	c_vectorizer = CountVectorizer(ngram_range = ngramrange, analyzer = 'char', vocabulary=featureset, dtype=np.int32)
	x_train_trans = c_vectorizer.fit_transform(train_corpus)
	x_dev_trans = c_vectorizer.transform(dev_corpus)
	x_test_trans = c_vectorizer.transform(test_corpus)


	le = LabelEncoder()
	y_train_trans = le.fit_transform(train_targets)
	y_dev_trans = le.transform(dev_targets)
	y_test_trans = le.transform(test_targets)

	
	outname = 'models/100feats_'+str(thisnumber)+'grams_'
	
	np.save(outname+'train_X', x_train_trans)
	np.save(outname+'train_y', y_train_trans)
	np.save(outname+'dev_X', x_dev_trans)
	np.save(outname+'dev_y', y_dev_trans)
	np.save(outname+'test_X', x_test_trans)
	np.save(outname+'test_y', y_test_trans)