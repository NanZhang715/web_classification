#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Feb 21 11:04:38 2019

@author: nzhang
"""

import pandas as pd
#import modin.pandas as pd  
import time
import re
import datetime
from sqlalchemy import create_engine
from utils.input_serve import fetch_mark, fetch_raw, migrate_file, getDomainShortNameByEnd 
from utils.feature_parse import feature_parse, read_snapshot, extract_title, extract_keywords, extract_description, extract_corpus 
from utils.input_utils import clean_text
from tensorflow.contrib import predictor
from serve import print_result
from utils.utils import Params, set_logger 
import os
import sys
import logging
import multiprocessing
import concurrent.futures
from functools import reduce
from utils.fetch_oss import OssOps


base_dir = 'serving/1552648936'
mark_record = './params/mark_record.json'
#positove folder
#local_folder ='/datanfs/rzx/toinner/p2p_snapshot/' 
local_folder ='/datanfs/rzx/p2p_snapshot/snapshot'


def predict_input_fn(tag_words, meta_words):
        
    for text, meta in zip(tag_words, meta_words):
        yield {'text':[text], 'meta':[meta]}


if __name__ == '__main__':

    #create log folder
    log_dir = './log'
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)
    
    start_time =datetime.datetime.now()
    timestamp = str(start_time)
    set_logger(os.path.join(log_dir,'{}_p2p_classifier.log'.format(timestamp)))
    
    sys.setrecursionlimit(10**8) 
    logging.info('The task starts at {}'.format(start_time))
    # 1. Obtain mark_id which stands for offset of last task
    mark_id = fetch_mark(mark_record)
    logging.info('The mark id is {}'.format(mark_id))

    # 2. Obtain raw data from Table Producer_snapshot
    # url, store_path, insert_time 
    
    inputs_sql = '''select
                      id as offset,
                      sname as website_name, 
                      domain,
                      url,
                     `snapshot` as file_path
               from 
                     producer_snapshot
               where  id > {} and snapshot is not NULL     
             ;'''.format(mark_id)
                      
    generator_df = fetch_raw(inputs_sql)
#   print('The shape of raw_data is {} '.format(generator_df.shape))
    
 
#    raw_data.columns
    for raw_data in generator_df:
        
        if raw_data.empty:
            logging.info('No new data, the task is terminated')
            continue
    
        # 3. record offset
        offset = int(raw_data['offset'].max())
        logging.info('The max id is {}'.format(offset))
       
        # 4. Parse store path of snapshot
        raw_data['process_time'] = time.strftime('%Y-%m-%d %H:%M:%S',time.localtime(time.time())) 
        snapshot_files =raw_data['file_path'].tolist()
        # 5. Extract feature from snapshot
        # COLUMNS: 
        #           id, website_name, domain, url, location,  process_time,
        #           title, keywords, description, corpus
        logging.info('Feature parse is on-going...')
        # read snapshots
        oss_ops = OssOps(
            oss_access_key_id = "vWkESI9cNqEyvLfx",
            oss_access_key_secret = "AfFCRvC5qAkskknswe8lewQtteRCFO",
            oss_bucket ="rzx-nfs",
            oss_endpoint = "http://oss-cn-beijing-jdm-d01-a.inner.certyun.cn/rzx-nfs" 
            )
    
        web_df = oss_ops.dataframe_oss(snapshot_files)
        #num_process = multiprocessing.cpu_count()-1
        #print(num_process)
        #with concurrent.futures.ProcessPoolExecutor(int(num_process)) as pool:
        #     web_df['title'] = list(pool.map(extract_title, web_df['unicode']))
        #     web_df['keywords'] = list(pool.map(extract_keywords, web_df['unicode']))
        #     web_df['description'] = list(pool.map(extract_description, web_df['unicode']))
        #     web_df['corpus'] = list(pool.map(extract_corpus, web_df['unicode']))

        features = feature_parse(web_df)
        data = raw_data.merge(features, how ='left',on='file_path')
        #data.to_csv('data.csv')
        logging.info('Feature parse is done')

        if data is None:
            logging.info('No snapshot files spotted, the task is terminated')
            continue
    
        # 6. Preprocess features
        # add corpus, meta columns
        tag_words, meta_words, data = clean_text(data)
        logging.info('Data contains {}'.format(data.columns))
        # drop columns
        data.drop(['offset','corpus', 'title', 'keywords', 'description','meta'], axis=1, inplace =True)
        logging.info('The preprocess dataframe contains  {}'.format(data.columns))
    
        
        # 7. Pipline for classify
        input_data = predict_input_fn(tag_words, meta_words)
        
        # 8. load model
        logging.info('model is loading ...')
        prediction_fn = predictor.from_saved_model(export_dir=base_dir)
        logging.info('model has loaded, print result in on-going...')
        result =print_result(list(map(prediction_fn,input_data)))
        logging.info('print result is done, the model result columns are {}'.format(result.columns))
                   
        # 9. keep positive data
        output_df = pd.concat([data, result], axis=1)
        output_df = output_df[output_df['classes']==1]
        output_df['logits']= output_df['logits'].map(lambda s: str(s) )
        output_df.drop('classes', axis=1 ,inplace=True)
        logging.info('{} positive data is exported to db'.format(output_df.shape[0]))
        logging.info('The Final columns are {}'.format(output_df.columns))
        output_df['top_domain'] = output_df['domain'].map(lambda s: getDomainShortNameByEnd(s))    
  
        # rename columns to make compromises with the stupid pingyin naming-culture 
        output_df.rename(
                columns={
                     'website_name':'wzmc',
                     'domain': 'ym',
                     'url': 'url',
                     'file_path':'nwdz',
                     'process_time': 'rksj',
                     'top_domain':'yjym',
                     'logits': 'logits'}, inplace = True)
      
        # 10. move positive sample to sampel folder
        target_files = output_df['nwdz'].tolist()
        # print(target_files)
        # print(type(target_files[0]))
        dst_dir = migrate_file(target_files, local_folder)
        logging.info('The {} snapshot files has been moved successfully'.format(len(target_files)))
        

        # 11. export to db 
        output_df['nwdz']= output_df['nwdz'].map(lambda s: os.path.join(dst_dir,s.split('/')[-1]))
  
        engine = create_engine("mysql+pymysql://rzx:rzx@1218!@!#@db-uikf.rzx.ifcert.cn:3308/funds_task?charset=utf8",encoding = 'utf-8')
        from sqlalchemy.dialects.mysql import BIGINT,VARCHAR, DATETIME, DOUBLE
        logging.info('The exported Dataframel columns are {}'.format(output_df.columns))
     
        output_df = output_df[['wzmc','ym','yjym','url','nwdz','rksj','logits']]
       
        dtype = {'wzmc': VARCHAR(255),
                 'ym': VARCHAR(255),
                  'yjym': VARCHAR(255),
                 'url': VARCHAR(255),
                 'nwdz': VARCHAR(255),
                 'rksj': DATETIME,
                 'logits':DOUBLE}
        logging.info('writing data to db ...')              
        pd.io.sql.to_sql(output_df,
                         name='yswz_p2p',
                         con=engine,
                         schema= 'funds_task',
                         if_exists='append',
                         index= False,
                         dtype = dtype)
        
        
        # 11. update offset_value
        record = Params(mark_record)
        record.offset_value = offset
        record.save(mark_record)
        logging.info('The offset value has been updated')
        logging.info('The updated offset value is {}'.format(record.offset_value))
         
        logging.info('_'*50)
    #hint of sucessful
    end_time = datetime.datetime.now()
    elapsed = (end_time-start_time).seconds 
    logging.info('Job is done at {}, used {} Seconds'.format(str(end_time), elapsed))    
    
    
    
    
