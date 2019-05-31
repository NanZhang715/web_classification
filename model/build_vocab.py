#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Created on Thu Jan 17 17:03:42 2019

@author: nzhang
"""
import gensim
import codecs

#PAD_WORD ='<PAD>'

def build_vocab(pre_trained_Word2vector):
    
    # load pre-trained Word2Vector
    model = gensim.models.KeyedVectors.load_word2vec_format(pre_trained_Word2vector,
                                                            binary= False)
    vocab_list = model.index2word
    
    return vocab_list

#pre_trained_Word2vector = '/home/nzhang/embed/Tencent_AILab_ChineseEmbedding.txt'
pre_trained_Word2vector = '/home/nzhang/embed/head.txt'
vocab_list =build_vocab(pre_trained_Word2vector)
    
with codecs.open('data/vocab.csv', 'w', encoding='utf-8') as vocab_file:
#        vocab_file.write("{}\n".format(PAD_WORD))
    for word in vocab_list:
        vocab_file.write("{}\n".format(word))

with codecs.open('data/nwords.csv', mode='w') as n_words:
    n_words.write(str(len(vocab_list)))
  
with codecs.open('data/vocab.tsv', 'w', encoding='utf-8') as vocab_file:
#        vocab_file.write("{}\n".format(PAD_WORD))
    for word in vocab_list:
        vocab_file.write("{}\n".format(word))


 
