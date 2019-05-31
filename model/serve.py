#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Feb 19 18:00:33 2019

@author: nzhang
"""

from tensorflow.contrib import predictor
from utils.input_utils import fetch_db
import pandas as pd

def predict_input_fn(sql):
    
    tag_words, meta_words = fetch_db(sql, is_predict=False)
    
    for text, meta in zip(tag_words, meta_words):
        yield {'text':[text], 'meta':[meta]}


def print_result(result):
    output = {}
    output['logits'] = []
    output['classes'] = []
    for res in result:
        output['classes'].append(int(res['classes']))
        output['logits'].append(res['probabilities'][0][int(res['classes'])])
    return pd.DataFrame.from_dict(output)


if __name__ == '__main__':
    
    #: a path to a directory containing a SavedModel
    base_dir = 'serving/1550568554'
    sql = "select title, keywords, description, corpus from p2p_df where  class= 'trainset' limit 100"
    
    prediction_fn = predictor.from_saved_model(export_dir=base_dir)

    data = predict_input_fn(sql)
    result =print_result(list(map(prediction_fn,data)))
    result.to_csv('result.csv')


