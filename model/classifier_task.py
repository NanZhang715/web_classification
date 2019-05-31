#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Feb 21 11:04:38 2019

@author: nzhang
"""

import pandas as pd   
import time
from sqlalchemy import create_engine
from utils.input_serve import fetch_mark, fetch_raw, migrate_file
from utils.feature_parse import feature_parse, parallel_read_snapshot
from utils.input_utils import clean_text
from tensorflow.contrib import predictor
from serve import print_result
from utils.utils import Params


base_dir = 'serving/1550574506'
mark_record = './params/mark_record.json'
#positove folder
#local_folder ='/datanfs/rzx/toinner/p2p_snapshot/' 
local_folder ='/datanfs/rzx/toinner/p2p_snapshots'

def predict_input_fn(tag_words, meta_words):
        
    for text, meta in zip(tag_words, meta_words):
        yield {'text':[text], 'meta':[meta]}


def main(argv=None):    
    
    # 1. Obtain mark_id which stands for offset of last task
    mark_id = fetch_mark(mark_record)
    print('The mark id is {}'.format(mark_id))

    # 2. Obtain raw data from Table Producer_snapshot
    # url, store_path, insert_time 
    
    inputs_sql = '''select 
                      id,
                      sname as website_name, 
                      domain,
                      url,
                     `snapshot` as file_path
               from 
                     tmp_producer_snapshot
               where  id > {} and snapshot is not NULL limit 100

             ;'''.format(mark_id)
                      
    raw_data = fetch_raw(inputs_sql)
    print('The shape of raw_data is {} '.format(raw_data.shape))
    
    if raw_data.empty:
        print('No new data, the task is terminated')
        return
 
    raw_data.columns
    
    # 3. record offset
    offset = raw_data['id'].max()
    print('The max id is {}'.format(offset))
   
    # 4. Parse store path of snapshot
    raw_data['process_time'] = time.strftime('%Y-%m-%d %H:%M:%S',time.localtime(time.time())) 
    snapshot_files =raw_data['file_path'].tolist()
    
    # 5. Extract feature from snapshot
    # COLUMNS: 
    #           id, website_name, domain, url, location,  process_time,
    #           title, keywords, description, corpus
    
    data = feature_parse(raw_data, snapshot_files)

    if data is None:
        print('No snapshot files spotted, the task is terminated')
        return

    # 6. Preprocess features
    # add corpus, meta columns
    tag_words, meta_words, data = clean_text(data)
    print('Data contains {}'.format(data.columns))
    # drop columns
    data.drop(['corpus', 'title', 'keywords', 'description','meta'], axis=1, inplace =True)
    print('The preprocess dataframe contains  {}'.format(data.columns))

    
    # 7. Pipline for classify
    input_data = predict_input_fn(tag_words, meta_words)
    
    # 8. load model
    prediction_fn = predictor.from_saved_model(export_dir=base_dir)
    result =print_result(list(map(prediction_fn,input_data)))
    print('The model result columns are {}'.format(result.columns))
    
   # data.to_csv('data.csv')
   # result.to_csv('result.csv')    
    
    # 9. export to db
    output_df = pd.concat([data, result], axis=1)
    output_df = output_df[output_df['classes']==1]
    output_df['logits']= output_df['logits'].map(lambda s: str(s) )
    output_df.drop('classes', axis=1 ,inplace=True)
    print('{} positive data is exported to db'.format(output_df.shape[0]))
    print('The Final columns are {}'.format(output_df.columns))

    # rename columns to make compromises with the stupid pingyin naming-culture 
    output_df.rename(
            columns={
                 'id':'id',
                 'website_name':'wzmc',
                 'domain': 'ym',
                 'url': 'url',
                 'file_path':'nwdz',
                 'process_time': 'rksj',
                 'logits': 'logits'}, inplace = True)
   
#    engine = create_engine("mysql+pymysql://root:Rzx@1218!@!#@202.108.211.109:51037/funds_task?charset=utf8",encoding = 'utf-8')
    engine = create_engine("mysql+pymysql://rzx:rzx@1218!@!#@db-uikf.rzx.ifcert.cn:3308/funds_task?charset=utf8",encoding = 'utf-8')
    from sqlalchemy.dialects.mysql import BIGINT,VARCHAR, DATETIME
    print('The exported Dataframel columns are {}'.format(output_df.columns))
 
    output_df = output_df[['id','wzmc','ym','url','nwdz','rksj','logits']]
   

    dtype = {'id':BIGINT,
             'wzmc': VARCHAR(200),
             'ym': VARCHAR(200),
             'url': VARCHAR(200),
             'nwdz': VARCHAR(200),
             'rksj': DATETIME,
             'logits': VARCHAR(200)}
                  
    pd.io.sql.to_sql(output_df,
                     name='p2p_classifer_output',
                     con=engine,
                     schema= 'funds_task',
                     if_exists='append',
                     index= False,
                     dtype = dtype)
    
    
    # 11. update offset_value
    record = Params(mark_record)
    record.offset_value = int(raw_data['id'].max())
    record.save(mark_record)
    print('The offset value has been updated')
    print('The updated offset value is {}'.format(record.offset_value))
     
    # 12. move positive sample to sampel folder
    target_files = output_df['nwdz'].tolist()
    # print(target_files)
    # print(type(target_files[0]))
    data = migrate_file(target_files, local_folder)
    print('The snapshot files has been moved successfully')
    
    #hint of sucessful
    print('Job is done')
    
if __name__ == '__main__':
    main()    
    
    
    
    
    
