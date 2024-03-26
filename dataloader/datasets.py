import numpy as np
import torch
from torch.utils import data
import pandas as pd
from transformers import BertTokenizer
import re
import jieba.posseg as pseg


import re
import jieba
from jieba import posseg as pseg

def word_slice(lines, stopwords=None):
    corpus = []
    corpus.append(lines.strip())
    stripcorpus = corpus.copy()
    for i in range(len(corpus)):
        stripcorpus[i] = re.sub("@([\s\S]*?):", "", corpus[i]) 
        stripcorpus[i] = re.sub("\[([\S\s]*?)\]", "", stripcorpus[i])
        stripcorpus[i] = re.sub("@([\s\S]*?)", "", stripcorpus[i])
        stripcorpus[i] = re.sub("[\s+\.\!\/_,$%^*(+\"\']+|[+——！，。？、~@#￥%……&*（）]+", "", stripcorpus[i])  
        stripcorpus[i] = re.sub("[^\u4e00-\u9fa5]", "", stripcorpus[i])
    
    onlycorpus = []
    for string in stripcorpus:
        if string == '':
            continue
        else:
            if len(string) < 5:
                continue
            else:
                onlycorpus.append(string)
    
    cutcorpusiter = onlycorpus.copy()
    cutcorpus = onlycorpus.copy()
    wordtocixing = []
    for i in range(len(onlycorpus)):
        cutcorpusiter[i] = pseg.cut(onlycorpus[i])
        cutcorpus[i] = ""
        for every in cutcorpusiter[i]:
            cutcorpus[i] = (cutcorpus[i] + " " + str(every.word)).strip()
            wordtocixing.append(every.word)
    
    if stopwords is not None:
        return [word for word in wordtocixing if word not in stopwords]
    else:
        return wordtocixing
    

class base_dataset(data.Dataset):
    def __init__(self, data_path):
        self.dataset = pd.read_csv(data_path,  delimiter='	')
        self.learning_map = {'餐馆':0,'景点':1,'酒店':2,'thank':3,'出租':4,'地铁':5,'没有意图':6,'bye':7,'greet':8}
        self.tokenizer = BertTokenizer.from_pretrained('./bert-base-chinese', do_lower_case=True)
        with open('./bert-base-chinese/cn_stopwords.txt', 'r', encoding='utf-8') as f:
            self.stopwords = [line.strip() for line in f.readlines()]

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        data = self.dataset.iloc[index]
        sentence, label = data['语料'], data['意图']
        sentence = word_slice(sentence, self.stopwords)
        sentence = ''.join(sentence)
        encoded_sentence = self.tokenizer.encode_plus(sentence, padding='max_length', truncation=True, max_length=32,return_tensors="pt")
        label = np.vectorize(self.learning_map.__getitem__)(label)
        return encoded_sentence['input_ids'], encoded_sentence['token_type_ids'], torch.tensor(label)
