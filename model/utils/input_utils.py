#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jan 17 15:05:28 2019

@author: nzhang
"""

import jieba_fast.analyse as jiebanalyse
import codecs
import jieba_fast as jieba
from sqlalchemy import create_engine
import pandas as pd
import re


def merge_columns(df):
    for data in df: 
        data['meta'] = data['title'].str.cat(data[["keywords", "description"]].astype(str))
    return data


def clean_text(data, stopwords_file='./stop_words.txt', topn=20):
    
    with codecs.open(stopwords_file, "r", "utf-8") as file:
        stop_words = [line.strip() for line in file.readlines()]
    
        
    data['corpus'] = data['corpus'].map(lambda s: re.sub('p2p','个贷',str(s),flags = re.IGNORECASE))
    data['corpus'] = data['corpus'].map(lambda s: re.sub(r'[^\u4E00-\u9FA5]','',str(s)))
    corpus = data['corpus'].tolist()
    tag_words = [[item for item in jiebanalyse.extract_tags(s, withWeight=False, topK=topn) if item not in stop_words] for s in corpus]
    tag_words = [' '.join(item) for item in tag_words]
    tag_words = [re.sub('个贷','p2p',str(item),flags = re.IGNORECASE) for item in tag_words]

    data['meta'] = data['title']+ data['keywords']+data['description']

    data['meta'].map(lambda s: re.sub('p2p','个贷', str(s),flags = re.IGNORECASE))
    data['meta'] = data['meta'].map(lambda s: re.sub(r'[^\u4E00-\u9FA5]','',str(s)))
    meta_words = data['meta'].tolist()
    meta_words = [[item for item in jieba.cut(item) if item not in stop_words] for item in meta_words]
    meta_words = [' '.join(item) for item in meta_words]
    meta_words = [re.sub('个贷','p2p',str(item),flags = re.IGNORECASE) for item in meta_words]

    return tag_words, meta_words, data
    
def fetch_db(sql, is_predict=True):
    """
    Loads data from files, splits the data into words and generates labels.
    Returns split sentences and labels.
    """
    engine = create_engine("mysql+pymysql://**********?charset=utf8",encoding = 'utf-8')
                           
    # Load data from files
    data = pd.read_sql(sql,con=engine)
    tag_words, meta_words, _  = clean_text(data)
   
    if is_predict: 
       label = list(data['label'].map(lambda s: int(s)))
       return tag_words, meta_words, label
    
    return tag_words, meta_words

