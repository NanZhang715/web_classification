#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jan 31 15:55:41 2019

@author: nzhang
"""

from tensorflow.contrib import predictor
import tensorflow as tf


#: a path to a directory containing a SavedModel
base_dir = 'serving/1550574506'

prediction_fn = predictor.from_saved_model(export_dir=base_dir)


predictions = prediction_fn({
    'text':
        ['常青藤 登录 p2p 平台 理财 投资 收益 盈宝 散标 网贷 账户 验证码 计算器 输入 行业资讯 用户 评论 优质 信息 网站'],
    'meta': 
        ['常青藤 专注 优质 抵押 资产 理财 信息 中介 平台 常青藤 理财网 贷钜石 理财 平台 网站 消费 金融 常青藤 官网 互联网 理财 常青藤 专注 优质 房产 抵押 综合 投资 理财 老 平台 用户注册 红包 加息 理财产品 日盈宝定 盈宝 隔日 计息 流转 灵活 投 门槛 低 随时 赎回 投资 理财 用户 选择 常青藤 官网 日盈宝定 盈宝 散标 购买 债权 方式 互联网 理财 稳健 收益']
})


    
print(predictions)
