#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue May 14 16:11:46 2019

@author: nzhang
"""

import os
import oss2
import pandas as pd
import time
from oss2 import exceptions


class OssOps(object):
    
    
    def __init__(self, 
               oss_access_key_id,
               oss_access_key_secret,
               oss_bucket,
               oss_endpoint):
        
        self.oss_access_key_id = oss_access_key_id,
        self.oss_access_key_secret = oss_access_key_secret,
        self.oss_bucket = oss_bucket,
        self.oss_endpoint = oss_endpoint
        
        self.auth = oss2.Auth(oss_access_key_id, 
                              oss_access_key_secret)
        
        self.bucket = oss2.Bucket(self.auth,
                                  oss_endpoint,
                                  oss_bucket)
        
    def read_oss(self, file):
        
        '''
        Return:
            strings  
        '''    
        return self.bucket.get_object(file).read().decode('utf-8')
    
    
    
    def dataframe_oss(self, file_path):
    
        '''
        read snapshot files [list]
    
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
                f = self.read_oss(file)
                data['unicode'].append(f)
                data['file_path'].append(file)
                    
        except (exceptions.NoSuchKey, exceptions.NoSuchBucket):
                print('{} is not exist'.format(file))
    
        return pd.DataFrame.from_dict(data)
        
    
    def upload_oss(self, key, content):
        
        '''
        Return:
            None
        '''    
        return self.bucket.put_object(key, content)
    
    
    def dir_list_oss(self):
        
        '''
        Print all bucket name
        '''
        service = oss2.Service(self.auth, self.oss_endpoint)
        print('\n'.join(info.name for info in oss2.BucketIterator(service)))
    

    def copy_oss(self, src_bucket, src_dir, dst_dir):
        
        '''
        Return:
            None
        '''    
        return self.bucket.copy_object(src_bucket, src_dir, dst_dir)
    
    
    def batch_copy_oss(self, key_list, dst_root):
        
        timestamp = time.strftime('%Y-%m-%d',time.localtime(time.time()))
        dst_dir = os.path.join(dst_root, timestamp)
        
        if not key_list:
            raise Exception('key_list should not be empty')
        
        mapping = {'share_disk': 'rzx-share', 'datanfs': 'rzx-nfs'}
        for key in key_list:
            self.copy_oss(
                          mapping[key.split('/')[0]],
                          '/'.join(key.split('/')[0:]),
                          os.path.join(dst_dir,key.split('/')[-1])
                          )
            
        return dst_dir

    

if __name__ == "__main__":
    
    oss_ops = OssOps(
            oss_access_key_id = "*******",
            oss_access_key_secret = "*******",
            oss_bucket ="*******",
            oss_endpoint = "*******" 
            )
    
    
    oss_file = 'datanfs/rzx/toinner/snapshot/B2694B9595C5E793BE30A2FDCDC53190'
    oss_ops.dir_list_oss()
    
    src_bucket = 'rzx-nfs'
    src_dir ="datanfs/rzx/toinner/snapshot/B2694B9595C5E793BE30A2FDCDC53190"
    dst_dir = "datanfs/rzx/p2p_snapshot/snapshot/2019-05-17/B2694B9595C5E793BE30A2FDCDC53190"
    oss_ops.copy_oss(src_bucket, src_dir, dst_dir)
    
    key_list = ['/datanfs/rzx/p2p_snapshot/snapshot/2019-05-16/CF4DE507626B984D2F13384633631294',
                '/datanfs/rzx/p2p_snapshot/snapshot/2019-05-16/CF4DE507626B984D2F13384633631294']
    dst_dir = 'datanfs/rzx/p2p_snapshot/snapshot/'
    
    key = key_list[1]
    mapping = {'share_disk': "rzx-share", 'datanfs': "rzx-nfs"}
    src_bucket =  mapping[key.split('/')[1]]
    src_dir = '/'.join(key.split('/')[2:]) 
    dst_dir = os.path.join(dst_dir,key.split('/')[-1])
    
    
    oss_ops.batch_copy_oss(key_list, dst_dir)
    
