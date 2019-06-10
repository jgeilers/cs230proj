import numpy as np
import tensorflow as tf
from tensorflow.python.layers.core import Dense
from distutils.version import LooseVersion
import warnings
import itertools
import collections
import os
import pickle
import copy
import problem_unittests as tests
import random
import re

from keras.preprocessing.text import Tokenizer
from sklearn.model_selection import train_test_split
from keras.preprocessing.sequence import pad_sequences
from keras.models import Sequential
from keras.layers import Embedding
from keras.layers import LSTM
from keras.layers import Dense
from keras.layers import RepeatVector
from keras.layers import TimeDistributed



def read_data(): 
    src_data  = open("./data/train.fr", "r")
    # new = open("fr_vocab", "w")
    src = []
    i = 0
    for line in src_data:
        if i % 100 == 0:
        	line = re.sub(r'[^a-zA-Z ]', "", line)
        	src.append(line)
            # new.write(line)
        i+=1
    # new.close()

    tgt_data = open("./data/train.en", "r")
    # new2 = open("en_vocab", "w")
    tgt = []
    i = 0
    for line in tgt_data:
        if i % 100 == 0:
        	line = re.sub(r'[^a-zA-Z ]', "", line)
        	tgt.append(line)
            # new2.write(line)
        i+=1
    # new2.close()
        
    return src, tgt

    #do this for the files and take all the bullshit out w REGEX


def split_data(X, Y): 
    x_train, x_dev, y_train, y_dev = train_test_split(X, Y, test_size=0.05, random_state=42)
    x_train, x_test, y_train, y_test = train_test_split(x_train, y_train, test_size = 0.05556, random_state = 42)


    # create vocabularies
    d = {}
    for line in y_train:
    	line = re.sub(r'[^a-zA-Z ]', "", line)
    	line = line.split(" ")
    	# print (line)
    	# print("hi")
    	
    	# line = line.split(" ")
    	# lines = re.sub(r'[^a-zA-Z ]', "", line)
    	# print(lines)
    	# print("hi")
    	for l in line:
	    	if l in d:
	    		d[l] += 1
	    	else:
	    		d[l] = 1

    d = sorted(d.items(), key=lambda kv: kv[1], reverse=True)
    print (d, len(d))


    # modify train/dev/test data to fit with model
    x = open("vocaben", "w")
    count = 0
    for item in d:
    	if count > 17000:
    		break
    	else:
    		s = item[0] + '\n'
    		x.write(s)
    	count += 1
    x.close()



    x = open("fr_train", "w")
    for line in x_train:
    	line = re.sub(r'[^a-zA-Z ]', "", line)
    	if line[:-1] == ".":
    		line = line[:-1] + " ." + '\n'
    	else:
    		line = line + " ." + '\n'
    	print(line)
    	x.write(line)
    x.close()

    y = open("en_train", "w")
    for line in y_train:
    	line = re.sub(r'[^a-zA-Z ]', "", line)
    	if line[:-1] == ".":
    		line = line[:-1] + " ." + '\n'
    	else:
    		line = line + " ." + '\n'
    	y.write(line)
    y.close()

    x1 = open("fr_test", "w")
    for line in x_test:
    	line = re.sub(r'[^a-zA-Z ]', "", line)
    	if line[:-1] == ".":
    		line = line[:-1] + " ." + '\n'
    	else:
    		line = line + " ." + '\n'
    	x1.write(line)
    x1.close()

    y1 = open("en_test", "w")
    for line in y_test:
    	line = re.sub(r'[^a-zA-Z ]', "", line)
    	if line[:-1] == ".":
    		line = line[:-1] + " ." + '\n'
    	else:
    		line = line + " ." + '\n'
    	y1.write(line)
    y1.close()

    x2 = open("fr_dev", "w")
    for line in x_dev:
    	line = re.sub(r'[^a-zA-Z ]', "", line)
    	if line[:-1] == ".":
    		line = line[:-1] + " ." + '\n'
    	else:
    		line = line + " ." + '\n'
    	x2.write(line)
    x2.close()

    y2 = open("en_dev", "w")
    for line in y_dev:
    	line = re.sub(r'[^a-zA-Z ]', "", line)
    	if line[:-1] == ".":
    		line = line[:-1] + " ." + '\n'
    	else:
    		line = line + " ." + '\n'
    	y2.write(line)
    y2.close()

    
    assert len(x_train) == len(y_train)
    assert len(x_dev) == len(y_dev)
    assert len(x_test) == len(y_test)
    return ((x_train, x_dev, x_test), (y_train, y_dev, y_test))

src, tgt = read_data()
x, y = split_data(src, tgt)
# print (x,y)

x_train, x_dev, x_test = x
x_train, x_dev, x_test = np.array(x_train), np.array(x_dev), np.array(x_test)
y_train, y_dev, y_test = y
y_train, y_dev, y_test = np.array(y_train), np.array(y_dev), np.array(y_test)

def initialize_tokenizer(text): 
    tokenizer = Tokenizer(filters='!"#$%&()*+,-./:;<=>?@[\\]^_`{|}~\t\n')
    tokenizer.fit_on_texts(text)
    return tokenizer

# initialize both tokenizers with respective datasets
french_tokenizer = initialize_tokenizer(src)
english_tokenizer = initialize_tokenizer(tgt)
# print(french_tokenizer)
# print("hi")
# print(english_tokenizer)

# calculate number of tokens in dataset
size_of_french_vocab = len(french_tokenizer.word_index)
size_of_eng_vocab = len(english_tokenizer.word_index)
# print('French vocab size: ', size_of_french_vocab)
# print('English vocab size: ', size_of_eng_vocab)
max_french_sentence_length = max([len(sentence) for sentence in src])
max_english_sentence_length = max([len(sentence) for sentence in tgt])
avg_french_length = sum([len(sentence) for sentence in src]) / len(src)



def pad_datasets(src, tgt): 
    src_padded = pad_sequences(sequences = src, maxlen = max_french_sentence_length, padding = 'post', dtype = "int32", truncating = 'post', value = -1)
    tgt_padded = pad_sequences(sequences = tgt, maxlen = max_english_sentence_length, padding = 'post', dtype = 'int32', truncating = 'post', value = -1)
    
    return src_padded, tgt_padded
    
src_sequence = french_tokenizer.texts_to_sequences(src)
tgt_sequence = english_tokenizer.texts_to_sequences(tgt)
# pad_datasets(src, tgt)
src_padded, tgt_padded = pad_datasets(src_sequence, tgt_sequence)
print("src padded length: ", len(src_padded))
print("tgt padded length: ", len(tgt_padded))







