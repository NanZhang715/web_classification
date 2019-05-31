#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Feb 21 10:55:49 2019

@author: nzhang
"""
from utils.utils import Params
import paramiko
import pandas as pd
import os
import time
from sqlalchemy import create_engine
import shutil
import codecs



def fetch_mark(mark_record):
    
    '''
    fetch record of offset
    '''
    record = Params(mark_record)
    mark_id = record.offset_value

#    record.offset_value = 5
#    record.save(mark_record)
    
    return mark_id

def fetch_raw(sql):
    
    '''
     obtain data from db
     @params:
         sql
    
    @return:
        Dataframe
    '''
    engine = create_engine("mysql+pymysql:/****************?charset=utf8",encoding = 'utf-8')
                 
    # Load data from files
    data = pd.read_sql(sql, con=engine, chunksize =2048)
    return data


def sftp_download_files(remotepath, localpath):
    
    '''
    download snapshot file from server
    '''
    try:
        t = paramiko.Transport(("***.***.***.***",******))
        t.connect(username = '****', password ='*******')
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
    
    
def migrate_file(src_file,dst_root):
    
    timestamp = time.strftime('%Y-%m-%d',time.localtime(time.time()))
    
    dst_dir = os.path.join(dst_root, timestamp)
    
    if not os.path.exists(dst_dir):
        os.makedirs(dst_dir)
    
    for file in src_file:
        dst_file = os.path.join(dst_dir, file.split('/')[-1])
        shutil.copyfile(file, dst_file)
    
    print('All files are copied')
            
    return dst_dir


topd = [".hb.cn" ,".net.cn" ,".com.cn" ,".yn.cn" ,".ha.cn" ,".he.cn" ,
        ".com.au" ,".nx.cn" ,".sd.cn" ,".sn.cn" ,".ah.cn" ,".bj.cn" ,
        ".gx.cn" ,".hl.cn" ,".gd.cn" ,".ac.cn" ,".gs.cn" ,".xj.cn" ,
        ".sc.cn" ,".ln.cn" ,".hk.cn" ,".zj.cn" ,".gov.cn" ,".edu.cn" ,
        ".cn.com" ,".hn.cn" ,".gz.cn" ,".js.cn" ,".com.hk" ,".sh.cn" ,
        ".fj.cn" ,".nm.cn" ,".net.au" ,".mo.cn" ,".tj.cn" ,".jx.cn" ,
        ".jl.cn" ,".qh.cn" ,".cq.cn" ,".org.cn" ,".xz.cn" ,".hi.cn" ,
        ".tw.cn" ,".com.tw" ,".sx.cn" ,".com.mo"]



def getDomainShortNameByEnd(domain):
    efix = ""
    for end in topd:
        if domain.endswith(end):
           domain = domain[:(len(domain)-len(end))]
           efix = end
           break
    if len(efix) == 0:
       try:
           index = domain.rindex(".")
           if index > 0:
               efix = domain[index:]
               domain = domain[:index]
       except:
        ValueError
    shortName = domain
    try:
        index = domain.rindex(".")
        if index >= 0:
            shortName = domain[index+1:]
    except:
        ValueError  
    return shortName + efix
