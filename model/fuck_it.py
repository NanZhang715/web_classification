#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Dec  4 09:36:47 2018

@author: nzhang
"""

import pandas as pd   
import pymysql
import paramiko
import codecs
import os
import re
import time
from bs4 import BeautifulSoup
from sqlalchemy import create_engine
from sshtunnel import SSHTunnelForwarder
import jieba_fast.analyse as jiebanalyse
from utils.input_serve import fetch_raw



def fetch_data_db(sql):

    ''' 
    IP/ICP Connect to the database  样本库

    '''

    connection = pymysql.connect(host='202.108.211.109',
                                 user='root',
                                 password='Rzx@1218!@!#',
                                 db='funds_task',
                                 charset='utf8mb4',
                                 port=51037,
                                 cursorclass=pymysql.cursors.DictCursor,
                                 connect_timeout=86400)

    try:
        with connection.cursor() as cursor:
            # Read a single record
            print(sql + ' is running')
            cursor.execute(sql)
            result = cursor.fetchall()
            result = pd.DataFrame(result)
            
    except ValueError:
        print("Error: Oopus, No Data, Drink a coffee")
        return None

    
    finally:
        connection.close()
        print(sql + 'is obtained')
        
    return result


def ssh_con_fetch_data(sql):
    
    '''
    
    SSH 代理登陆 MYSQL 服务器
    
    '''
    
    with SSHTunnelForwarder(
            ('202.108.211.109',51007),  ## 堡垒机地址
            ssh_username = 'root',
            ssh_password = 'rzx@1218!@!#',
            remote_bind_address = ('10.130.21.150',3308)  ## mysql服务器地址
            ) as server:
        
        print(server.local_bind_port)       
        
        # Connect to the database
        connection = pymysql.connect(host='127.0.0.1',  ##必须是127.0.0.1 
                                     user='rzx',
                                     password='rzx@1218!@!#',
                                     db='funds_task',
                                     charset='utf8mb4',
                                     port=server.local_bind_port,
                                     cursorclass=pymysql.cursors.DictCursor,
                                     connect_timeout=86400)
    
        try:
            with connection.cursor() as cursor:
                # Read a single record
                print(sql + ' is running')
                cursor.execute(sql)
                result = cursor.fetchall()
                result = pd.DataFrame(result)
                
        except ValueError:
            print("Error: Oopus, No Data, Drink a coffee")
            return None
       
        finally:
            connection.close()
            print(sql + 'is obtained')
            
        return result


def sftp_download_files(remotepath, localpath):
    try:
        t = paramiko.Transport(("202.108.211.109",51007))
        t.connect(username = 'root', password ='rzx@1218!@!#')
        sftp = paramiko.SFTPClient.from_transport(t)
        
    #remotepath='/datanfs/rzx/toinner/snapshot'
    #localpath='/tmp/system.log'  
    
        sftp.get(remotepath, localpath)
        print('{} is obtained successfully'.format(remotepath.split('/')[-1]))
        
    except IOError:
         print('{} is not exists'.format(remotepath.split('/')[-1]))
        
    finally:
        t.close()
        
        return
    
    
def read_snapshot(file_path):
    
    data = {}
    data['unicode'] = []
    data['file_name'] = []
    for file in file_path:
        try:
            with codecs.open(file, "r", "utf-8")as f:
                data['unicode'].append(str(f.read()))
                data['file_name'].append(file.split('/')[-1])
        except:
            print('{} is not exist !'.format(file))
            
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
    

if __name__ == '__main__':
   
#   file_name = 'new_sample.xlsx'
#   data = pd.read_excel(file_name,sheet_name= 'Sheet2')
#   
#   url = (','.join('"'+item +'"' for item in data['网站'].tolist())) 
#       sql = '''select  
#                   wzmc as website_name,
#                   wwsywz as url,
#                   xcsywz as store_path                  
#            from
#                qrjrwz 
#            where 
#                wwsywz in ({}) 
#            and
#                xcsywz is not null'''.format(url)
    
   sql  =  '''
            select 
                    url as url, 
                    nwdz as location
        
            from 
                    ybwz_jysyt 
            where fwzt = 1
            '''
            
                                               
   df_db = fetch_raw(sql)
   for data in df_db:
       data['file_name'] = data['location'].map(lambda s : s.split('/')[-1])
        
       print('The shape of output is {} '.format(data.shape))
       
       files_path =data['location'].tolist()     
       df = read_snapshot(files_path)  # unicode, file_name  
     
       df['corpus'] = df['unicode'].map(lambda s: re.sub('p2p','个贷',str(s),flags = re.IGNORECASE))
       df['corpus'] = df['corpus'].map(lambda s: re.sub(r'[^\u4E00-\u9FA5]','',str(s)))
       df['corpus'] = df['corpus'].map(lambda s: s.replace('个贷','p2p'))
       
       df['title'] = df['unicode'].map(lambda s :extract_title(s))
       df['keywords'] = df['unicode'].map(lambda s : extract_keywords(s))
       df['description'] = df['unicode'].map(lambda s : extract_description(s))
       
       df = df.merge(data,how ='left',on='file_name')
      
       df.drop('file_name',axis =1, inplace =True)
       df.drop('location',axis=1, inplace=True)
       print(df.columns)
      
      # engine = create_engine("mysql+pymysql://root:Rzx@1218!@!#@202.108.211.109:51037/funds_task?charset=utf8mb4",encoding = 'utf-8')
       engine = create_engine("mysql+pymysql://root:Rzx@1218!@!#@10.130.21.67/funds_task?charset=utf8mb4",encoding = 'utf-8')       
 
       from sqlalchemy.dialects.mysql import LONGTEXT, INTEGER, VARCHAR, TEXT
        
       dtypedict = { 'url':VARCHAR(100),
                    'website_name':VARCHAR(100),
                    'unicode':LONGTEXT,
                    'corpus':LONGTEXT,
                    'title':TEXT,
                    'keywords':TEXT,
                    'description':TEXT}
                      
       pd.io.sql.to_sql(df,
                         name='jys_feedback',
                         con=engine,
                         schema= 'funds_task',
                         if_exists='append',
                         index= False,
                         dtype = dtypedict)
