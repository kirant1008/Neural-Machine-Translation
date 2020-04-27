#Importing all the required libraries
import pandas as pd
import preprocess as pp

import argparse

import torch
import torch.functional as F
import torch.nn as nn
import torch.optim as optim
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence

from sklearn.model_selection import train_test_split
import numpy as np
import unicodedata
import re
import time


import warnings
warnings.filterwarnings("ignore")

import nltk

#Defining tokenizers as a function
tokenizers = {
	    'en': nltk.tokenize.WordPunctTokenizer().tokenize,
	    'vi': nltk.tokenize.WordPunctTokenizer().tokenize
	}

import nmt_model as nmt
from tqdm import tqdm

#Defining start and end token
SOS_token = '<start>'
EOS_token = '<end>'
UNK_token = '<unk>'
PAD_token = '<pad>'
#Defining start stop for both english and vietnamese
SOS_idx = 0
EOS_idx = 1
UNK_idx = 2
PAD_idx = 3

#Data Preprocessing
source_lang = 'en'
target_lang = 'vi'

import nltk
import pandas as pd

#Defining max length variable to take sentences only of maximum length 25
max_len = 25
min_word_count = 1

#Loading all the data from the url
print("Loading Data................\n")
url_en = 'https://nlp.stanford.edu/projects/nmt/data/iwslt15.en-vi/train.en'
url_vi = 'https://nlp.stanford.edu/projects/nmt/data/iwslt15.en-vi/train.vi'
text_en = pd.read_csv(url_en,'\n',header = None)
text_vi = pd.read_csv(url_vi,'\n',header = None)
data = pd.concat([text_en,text_vi],axis=1)
data.columns = ["source","target"]
#Due to some string error taking values in .notnull()
data = data[data["source"].notnull()]
data = data[data["target"].notnull()]


#Send this data in preprocess_sentence where sentence words are lowered and unwanted characters are removed
data["source"] = data.source.apply(lambda w: pp.preprocess_sentence(w))
data["target"] = data.target.apply(lambda w: pp.preprocess_sentence(w))

#Keeping sentences which are less than max_len = 25
# data = data[data["target"].str.split(" ").str.len() <= max_len]
# data = data[data["source"].str.split(" ").str.len() <= max_len]
data = data[(data['source'].str.split(" ").str.len() <= max_len) & (data['target'].str.split(" ").str.len() <= max_len)]
data = data.reset_index().drop('index',1)

#Loading  test Data
url_en_test = 'https://nlp.stanford.edu/projects/nmt/data/iwslt15.en-vi/tst2012.en'
url_vi_test = 'https://nlp.stanford.edu/projects/nmt/data/iwslt15.en-vi/tst2012.vi'
text_en_test = pd.read_csv(url_en_test,'\n',header = None)
text_vi_test = pd.read_csv(url_vi_test,'\n',header = None)

data_test = pd.concat([text_en_test,text_vi_test], axis = 1)
data_test.columns = ['source','target']

#Send this data in preprocess_sentence where sentence words are lowered and unwanted characters are removed
data_test["source"] = data_test.source.apply(lambda w: pp.preprocess_sentence(w))
data_test["target"] = data_test.target.apply(lambda w: pp.preprocess_sentence(w))

# #Keeping sentences which are less than max_len = 25
# data_test = data_test[data_test['target'].str.split(" ").str.len() <= max_len]
# data_test = data_test[data_test['source'].str.split(" ").str.len() <= max_len]
data_test = data_test[(data_test['source'].str.split(" ").str.len() <= max_len) & (data_test['target'].str.split(" ").str.len() <= max_len)]
data_test = data_test.reset_index().drop('index',1)

# Concatenating train and test data
entire_data = pd.concat([data,data_test],axis =0)

#Tokenizing the sentences using tokenizer, 
src_sents = pp.preprocess_func(entire_data['source'], tokenizers[source_lang], min_word_count)
trgt_sents = pp.preprocess_func(entire_data['target'], tokenizers[target_lang], min_word_count)

#Generating Source and Target Vocab from Training Data
src_vocab = pp.read_vocab(src_sents)
trgt_vocab = pp.read_vocab(trgt_sents)

#Spliting the data in train and test indices
training_indices = np.arange(0,19431)
test_indices = np.arange(19431,19947)

#Generating preprocessed training and test data with the help of indices
training_source = [src_sents[i] for i in training_indices]
test_source = [src_sents[i] for i in test_indices]

training_target = [trgt_sents[i] for i in training_indices]
test_target = [trgt_sents[i] for i in test_indices]

#Adding 0's to the tensors that is padding according to max length
max_seq_length = max_len + 2  # 2 for EOS_token and SOS_token

#Now we generate tensors with the help of tsr_pr func
training = []
for source_sent, target_sent in zip(training_source, training_target):
    training.append(pp.tensrs_prs(src_vocab,trgt_vocab,source_sent, target_sent, max_seq_length))


#Dividing into source and target
x_training, y_training = zip(*training)

x_training = torch.transpose(torch.cat(x_training, dim=-1), 1, 0)
y_training = torch.transpose(torch.cat(y_training, dim=-1), 1, 0)

testing = []
for source_sent_test,target_sent_test in zip(test_source, test_target):
    testing.append(pp.tensrs_prs(src_vocab,trgt_vocab,source_sent_test, target_sent_test, max_seq_length))


x_test, y_test = zip(*testing)

x_test = torch.transpose(torch.cat(x_test, dim=-1), 1, 0)
y_test = torch.transpose(torch.cat(y_test, dim=-1), 1, 0)


def train(source_lang = 'en',target_lang = 'vi',x_training = x_training,y_training = y_training,source_vocab = src_vocab, target_vocab = trgt_vocab):

	# Printing Training Data Information
	print('Data length: {}\nSource vocabulary size: {}\nTarget vocabulary size: {}\n'.format(
	    len(src_sents), len(src_vocab.word2index), len(trgt_vocab.word2index)
	))
	examples = list(zip(src_sents, trgt_sents))[10:15]
	for source, target in examples:
	    print('Source: "{}", Target: "{}"'.format(' '.join(source), ' '.join(target)))

	from torch.optim import Adam
	import nmt_model as nmt
	
	#Initializing the model
	model = nmt.seq2seq(len(source_vocab), len(target_vocab), 1024, 1)
	model = model.cuda()
	#Defining the Adam optimizer
	optim = Adam(model.parameters(), lr=0.0001)


	#Sending the training data to gpu
	x_training = x_training.cuda()
	y_training = y_training.cuda() 


	#Defining Cross Entropy Loss
	cross_entropy = nn.CrossEntropyLoss()


	#Training the model
	from tqdm import tqdm

	BATCH_SIZE = 64
	total_batches = int(len(x_training)/BATCH_SIZE) + 1
	indices = list(range(len(x_training)))

	print("\nStart Model Training.........................................\n")

	loss_prev = []

	for epoch in range(40):
	    # Training
	    total_loss = 0.0
	    #generating the text data in to batches and feeding it to the model
	    for step, batch in tqdm(enumerate(pp.batch_generator(indices, BATCH_SIZE)),
                            desc='Training epoch {}'.format(epoch+1),
                            total=total_batches):

	        x = x_training[batch, :]
	        # y for teacher forcing is all sequence without a last element
	        y_teachf = y_training[batch, :-1]
	        # y for loss calculation is all sequence without a last element
	        y_true = y_training[batch, 1:]
	        # (batch_size, vocab_size, seq_length)
	        H = model.forward_train(x, y_teachf)
	        loss = cross_entropy(H, y_true)

	        assert loss.item() > 0

	        optim.zero_grad()
	        loss.backward()
	        optim.step()

	        total_loss += loss.item()

	    loss_prev.append(total_loss/total_batches)
	    print('Epoch {} training is finished, loss: {:.4f}'.format(epoch+1, total_loss/total_batches))

	print("\nModel Training Finished")
	torch.save(model.state_dict(), './model/mdl_weights.pth')

def test(source_lang = 'en',target_lang = 'vi',x_test = x_test,source_vocab = src_vocab, target_vocab = trgt_vocab, test_target = test_target, test_source = test_source):

	print("Testing Model : ")
	from nltk.translate.bleu_score import corpus_bleu, SmoothingFunction

	def bleu(n,list_of_references,list_of_hypothesis):
	    cc = SmoothingFunction()
	    weights = [1.0/n]*n + [0.0]*(4-n)
	    score = corpus_bleu(list_of_references, list_of_hypothesis,weights,smoothing_function = cc.method1)
	    return score
	    
	def testing(model, X, target, desc='testing...'):
	    length = len(target)
	    list_of_hypothesis = []
	    for i, x in tqdm(enumerate(X),
	                      desc=desc,
	                      total=length):
	        y = model(x.unsqueeze(0))
	        hypothesis = target_vocab.unidex_words(y[1:-1])  # Remove SOS and EOS from y
	        list_of_hypothesis.append(hypothesis)
	    score = bleu(1,target, list_of_hypothesis)
	    return score
	#sending x_test to gpu
	x_test = x_test.cuda()

	#initializing the model
	net = nmt.seq2seq(len(source_vocab),len(target_vocab),1024,1)
	net = net.cuda()
	net.load_state_dict(torch.load('./model/mdl_weights.pth'))

	#getting the bleu score
	test_scores = testing(net, x_test, test_target)
	print('\nAverage BLEU score:',test_scores)

	examples = zip(test_source[11:21], y_test[11:21], x_test[11:21])

	print('\nTest Translation : ')
	for source, target, x in examples:
	    y = net(x.unsqueeze(0))
	    translation = ' '.join(target_vocab.unidex_words(y[1:-1]))
	    source = ' '.join(source)

	    print('Source: "{}"\nTranslation: "{}"\n'.format(source, translation))

def translate(x_test = x_test,y_test = y_test,target_vocab = trgt_vocab,source_vocab = src_vocab):
	#we will be able to translate sentence untill we give keyboard interrupt
	print("Enter Sentences for translation : ")
	while(True):
		sentence = input("Source : ")
		#preprocessing the sentence
		sentence1 = pp.preprocess_sentence(sentence)
		#tokenizing the sentence
		sentence1 = pp.preprocess_corpus_translate(sentence1, tokenizers['en'])
		#converting it to tensor
		sentence_tensor = pp.tensors_from_pair_translate(source_vocab,sentence1[0],max_seq_length)
		sentence_tensor = torch.transpose(sentence_tensor, 1, 0)

		#initizaling the model
		net = nmt.seq2seq(len(source_vocab),len(target_vocab),1024,1)
		net = net.cuda()
		net.load_state_dict(torch.load('./model/mdl_weights.pth'))

		#sending the sentence to cuda
		sentence_tensor = sentence_tensor.cuda()
		y = net(sentence_tensor)
		translation = ' '.join(target_vocab.unidex_words(y[1:-1]))
		print('Translation: "{}"\n'.format(translation))

#using argparse to give commandline commands

parser1 = argparse.ArgumentParser()
parser1.add_argument("train", nargs="?", default=None, help="call the traininig function")
parser2 = argparse.ArgumentParser()
parser2.add_argument("test", nargs="?", default=None, help="calling test function")
parser3 = argparse.ArgumentParser()
parser3.add_argument("translate", nargs="?", default=None, help="calling trasnlate function")

args1 = parser1.parse_args()
args2 = parser2.parse_args()
args3 = parser3.parse_args()

if args1.train == 'train' :
    train()
if args2.test == 'test' :
    test()
if args3.translate == 'translate' :
    translate()


