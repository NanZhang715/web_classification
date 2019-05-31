#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Feb 21 11:24:34 2019

@author: nzhang
"""
import pandas as pd 
#import modin.pandas as pd  
import codecs
import re
from bs4 import BeautifulSoup
import multiprocessing
import numba


def read_snapshot(file_path):
    
    '''
    read snapshot files 
    
    @params:
        file_path: store path of snapshot
        
    @return:
        DataFrame 
    '''
    
    data = {}
    data['unicode'] = []
    data['file_path'] = []
    try:
        for file in file_path:
            with codecs.open(file, "r", "utf-8") as f:
                 data['unicode'].append(f.read())
                 data['file_path'].append(file)
    except FileNotFoundError:
            print('{} is not found'.format(file))
            
    return pd.DataFrame.from_dict(data)


def extract_title(html_doc):
    
    try:
        soup = BeautifulSoup(html_doc, 'html.parser')
        title = soup.title.string
        
    except:       
        title = None
    
    return title


def extract_keywords(html_doc):
    
    try:
        soup = BeautifulSoup(html_doc, 'html.parser')
        keywords = soup.find(attrs={"name": re.compile('keywords',re.IGNORECASE)})['content']

        
    except:       
        keywords = None
    
    return keywords


def extract_description(html_doc):
    
    try:
        soup = BeautifulSoup(html_doc, 'html.parser')
        description = soup.find(attrs={"name":re.compile('description',re.IGNORECASE)})['content']
        
    except:
        description = None
    
    return description    

def extract_corpus(html_doc):

    return re.sub(r'[^\u4E00-\u9FA5]','',str(html_doc))



def feature_parse(web_df):
    '''
    extract fearures from snapshot, and merge with website basic info
    
    @params:
        data: dataframe from producer_snapshot table
        file_name: the store path of snaptshot 
    
    @return:
        DataFrame
    '''
    #web_df = read_snapshot(file_list)
    #if web_df.empty:
    #    return None
    #print('{} rows in DataFrame'.format(web_df.shape[0]))

    # Extract features
    web_df['title'] = web_df['unicode'].map(lambda s :extract_title(str(s)))
    web_df['keywords'] = web_df['unicode'].map(lambda s : extract_keywords(s))
    web_df['description'] = web_df['unicode'].map(lambda s : extract_description(s))
    web_df['corpus'] = web_df['unicode'].map(lambda s: re.sub(r'[^\u4E00-\u9FA5]','',str(s)))
    
    # remove unicode column
    web_df.drop('unicode', axis=1, inplace =True)
    
    #df = data.merge(web_df, how ='left',on='file_path')
    
    return web_df


def parallel_read_snapshot(file_path):
     
    n_cpu = multiprocessing.cpu_count()
    pool = multiprocessing.Pool(n_cpu)
    
    result = pool.map(read_snapshot,file_path)
    pool.close()
    pool.join()
           
    return result





  
