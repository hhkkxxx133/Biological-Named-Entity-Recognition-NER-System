import os
import numpy as np
import random
import sklearn.svm
import sklearn.linear_model
import nltk
import time

greek = ['alpha', 'beta', 'gamma', 'delta', 'epsilon', 'eta', 'theta', 
		 'mu', 'xi', 'iota', 'lambda', 'kappa', 'zeta']

stop_words = ['or', 'and', 'in', 'to', 'of', 'as', 'the', 'a', 'an', 
			  'not', 'that', 'such', 'by', 'be', 'is', 'are', 'at', 
			  'for', 'from', 'has', 'he', 'his', 'she', 'her', 'they', 
			  'their', 'it', 'its', 'on', 'was', 'were', 'will', 'with', 
			  'so', 'if']

def generate_feature(word, tag, min_dis):
	feature = np.zeros(8)
	contain_num = False
	contain_alpha = False
	for letter in word:
		if letter.isalpha():
			contain_alpha = True
		elif letter.isnumeric():
			contain_num = True
		if contain_num and contain_alpha:
			feature[3] = 1 # Alphanumeric
			break
	if word.isupper() and len(word) > 1:
		feature[0] = 1 # All capital letter
	elif contain_alpha and not word.islower():
		feature[1] = 1 # Mixed upperclass and lowerclass
	if ('-' in word or '+' in word or '/' in word) and len(word) > 1:
		feature[2] = 1 # Contains '-' or '+'
	if word[0].isupper() and len(word) > 1:
		feature[4] = 1 # Initial capitals
	for gletter in greek:
		if gletter in word.lower() and len(word) > len(gletter):
			feature[5] = 1 # Greek Alphabet
			break
	if 'NN' in tag:
		feature[6] = 1 # Part-of-Speech tagger
	feature[7] = min_dis
	return list(feature)

def levenshtein_dis(word1, word2):
	if min(len(word1), len(word2)) == 0:
		return max(len(word1), len(word2))
	cur_dis = 0
	if word1[len(word1) - 1] == word2[len(word2) - 1]:
		cur_dis = 0
	else:
		cur_dis = 1
	return min(levenshtein_dis(word1[:-1], word2) + 1, levenshtein_dis(word1, word2[:-1]) + 1, levenshtein_dis(word1[:-1], word2[:-1]) + cur_dis)

def similarity(test, train):
	train_words = train.split()
	max_sim = -1
	for train_word in train_words:
		sim = 0
		for i in range(min(len(test), len(train_word))):
			if test[i] == train_word[i]:
				sim += 1
			else:
				break
		for i in range(1, min(len(test), len(train_word)) + 1):
			if test[-i] == train_word[-i]:
				sim += 1
			else:
				break
		max_sim = max(sim, max_sim)
	score = max_sim / len(test)
	if score >= 1:
		score = score / 2
	return score

def accuracy(path, predict):
	path = path + 'a1'
	test_dic = dict()
	for line in open(path, 'r').read().splitlines():
		cur_protein = line.split('\t')[2]
		if cur_protein not in test_dic:
			test_dic[cur_protein] = 1
	TP = 0
	FP = 0
	FN = 0
	for predict_protein in predict:
		if predict_protein in test_dic:
			TP += 1
		else:
			FP += 1
	for test in test_dic:
		if test not in predict:
			FN += 1
	if len(test_dic) == 0:
		return -1, -1
	else:
		recall = TP / (TP + FN)
	if len(predict) == 0:
		precision = 0
	else:
		precision = TP / (TP + FP)
	return precision, recall


class NER():
	def __init__(self, clf):
		self.classifier = clf
		self.dic = dict()
		
	def train(self, paths):
		X = []
		for path in paths:
			os.chdir(path)
			filenames_list = os.listdir()
			for filename in filenames_list:
				if filename[-2:] == 'a1':
					for line in open(filename,'r').read().splitlines():
						cur_protein = line.split('\t')[2]
						if cur_protein not in self.dic and not cur_protein.isdigit():
							self.dic[cur_protein] = 1
			for filename in filenames_list:
				if filename[-3:] == 'txt':
					file_array = open(filename,'r').read().split()
					clean_array = list(map(lambda x : x.strip(';,.()'), file_array))
					part_of_speech_tag = dict(nltk.pos_tag(clean_array))
					for word in clean_array: 
						if len(word) > 0:
							max_dis = 0
							is_protein = False
							for protein in self.dic:
								if word in protein:
									is_protein = True
								cur_dis = similarity(word, protein)
								if cur_dis > 1 and cur_dis < 2:
									cur_dis = cur_dis / 2
								max_dis = max(cur_dis, max_dis)
							vector = generate_feature(word, part_of_speech_tag[word], max_dis)
							if is_protein: 
								X.append(vector + [1]) # 1 indicates the word is a protein
							else:
								X.append(vector + [0]) # 0 indicates the word is NOT a protein
		random.shuffle(X)
		X = np.array(X)
		self.classifier.fit(X[:, :-1], X[:, -1])
	
	def predict(self, path):
		os.chdir(path)
		filenames_list = os.listdir()
		protein_predict = []
		file_pre = ''
		precision_list = []
		recall_list = []
		for filename in filenames_list:
			if filename[-3:] == 'txt':
				file_pre = filename[:-3]
				protein_list = []
				file_array = open(filename,'r').read().split()
				clean_array = list(map(lambda x : x.strip(';,.()\''), file_array))
				clean_array = list(filter(None, clean_array))
				part_of_speech_tag = dict(nltk.pos_tag(clean_array))
				to_append = []
				for word in clean_array:
					if len(word) > 0 and not word.isdigit() and (word.lower() == 'and' or word.lower() == 'of' or word.lower() not in stop_words):
						max_dis = 0
						for protein in self.dic:
							if word in protein:
								max_dis = 2
								break
							cur_dis = similarity(word, protein)
							max_dis = max(cur_dis, max_dis)
						vector = np.array(generate_feature(word, part_of_speech_tag[word], max_dis))
						if self.classifier.predict(vector.reshape(1, -1)):
							protein_list.append(word)
						# if word.lower() == 'and' or word.lower() == 'of':
						# 	if len(to_append) > 0:
						# 		to_append.append(word)
						# elif self.classifier.predict(vector.reshape(1, -1)):
						# 	if len(to_append) > 0:
						# 		if to_append[-1] == 'and' and 'NN' in part_of_speech_tag[word]:
						# 			to_append.pop(-1)
						# 			protein_list.append(' '.join(to_append))
						# 			to_append = []
						# 	to_append.append(word)
						# elif len(to_append) > 0:
						# 	if to_append[-1].lower() == 'and' or to_append[-1].lower() == 'of' or (len(to_append) == 1 and len(to_append[-1]) == 1):
						# 		to_append.pop(-1)
						# 	if len(to_append) > 0:
						# 		protein_list.append(' '.join(to_append))
						# 		to_append = []
				precision, recall = accuracy(file_pre, protein_list)
				if precision < 0:
					continue
				else:
					precision_list.append(precision)
					recall_list.append(recall)
				protein_list.append(filename)
				protein_list.append(precision)
				protein_list.append(recall)
				protein_predict.append(protein_list)
		return protein_predict, precision_list, recall_list


def main():
	clf_svm = sklearn.svm.SVC()
	clf_percp = sklearn.linear_model.Perceptron()
	my_ner = NER(clf_svm)
	start = time.time()
	#my_ner.train(['./BioNLP-ST-2013_GE_train_data_rev3'])
	my_ner.train(['./BioNLP-ST-2013_GE_train_data_rev3', '../BioNLP-ST-2013_GE_devel_data_rev3'])
	print(time.time() - start)
	start = time.time()
	print('predicting')
	outcome, precision, recall = my_ner.predict('../BioNLP-ST-2013_GE_test_data_rev1')
	print(precision.count(0))
	print(recall.count(0))
	P = np.mean(precision)
	R = np.mean(recall)
	F = (2 * P * R) / (P + R)
	print(P)
	print(R)
	print(F)
	np.save('outcome', outcome)
	print(time.time() - start)


if __name__ == "__main__":
	main()
