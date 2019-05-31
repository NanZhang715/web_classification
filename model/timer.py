#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Feb 27 16:23:27 2019

@author: nzhang
"""

import time
import pymysql.cursors
from utils.utils import Params
from utils.input_serve import fetch_mark


timestamp = time.strftime('%Y_%m_%d',time.localtime(time.time()))
mark_path = './params/mark_record.json'
sql_rename = 'CREATE TABLE yswz_p2p_{}_bak SELECT * FROM yswz_p2p;'.format(timestamp)

mark_record = './params/mark_record.json'
mark_id = fetch_mark(mark_record)
sql_trunc = 'DELETE FROM yswz_p2p where id < {};'.format(mark_id)


def reset_offset(mark_path):
    
    record = Params(mark_path)
    mark_id = record.offset_value
    
    record.offset_value = 0
    record.save(mark_path)
    
    print('Offset value has been set from {} to 0'.format(mark_id))
    return


def rename_table(sql_rename):
    
    connection = pymysql.connect(host='db-uikf.rzx.ifcert.cn',
                             user='rzx',
                             password='rzx@1218!@!#',
                             db='funds_task',
                             charset='utf8mb4',
                             port= 3308,    
                             cursorclass=pymysql.cursors.DictCursor)
    try:
        with connection.cursor() as cursor:
            # Create a new record
            cursor.execute(sql_rename)
    
        # connection is not autocommit by default. So you must commit to save
        # your changes.
        connection.commit()
            
    finally:
        connection.close()
    
    return


def delete_table(sql_trunc):
    
    connection = pymysql.connect(host='db-uikf.rzx.ifcert.cn',
                         user='rzx',
                         password='rzx@1218!@!#',
                         db='funds_task',
                         charset='utf8mb4',
                         port= 3308,    
                         cursorclass=pymysql.cursors.DictCursor)
        
    try:
        with connection.cursor() as cursor:
            # Create a new record
            cursor.execute(sql_trunc)
    
        # connection is not autocommit by default. So you must commit to save
        # your changes.
        connection.commit()
            
    finally:
        connection.close()
    
    return

    
def main(argv=None): 
    
    rename_table(sql_rename)
    delete_table(sql_trunc)
    reset_offset(mark_path)
    
    return
    
    
if __name__ == '__main__':
    main()
        