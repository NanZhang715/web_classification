#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jan 17 11:21:25 2019

@author: nzhang
"""

import warnings
import tensorflow as tf
from utils.input_utils import fetch_db
from utils import utils
import os
import numpy as np
import logging
import gensim
import multiprocessing

warnings.filterwarnings("ignore")

flags = tf.app.flags

flags.DEFINE_integer("num_classes", 2, "the num of classes")
flags.DEFINE_integer("train_steps", 50000, "Number of (global) training steps to perform")
flags.DEFINE_integer("meta_max_len", 80, "max length of sentences")
flags.DEFINE_integer("body_max_len", 20, "max length of sentences")
flags.DEFINE_string("model_dir", './output', "Base directory for the model")


tf.flags.DEFINE_string("stops_words", './stop_words.txt', "the file contains stop words")
tf.flags.DEFINE_string("pre_trained_Word2vector", "/home/nzhang/embed/Tencent_AILab_ChineseEmbedding.txt", "Data source for Word2vector.")
tf.flags.DEFINE_string("path_vocab", "./data/vocab.csv", "word table")
tf.flags.DEFINE_integer("embedding_dim", 200, "Dimensionality of character embedding")
tf.flags.DEFINE_integer("num_oov_buckets",0 , "the number of out of bag vocab")
tf.flags.DEFINE_string("sql_train", "select title, keywords, description, corpus, label from p2p_corpus where  class= 'trainset'", "SQL querys trainset")
tf.flags.DEFINE_string("sql_test", " select title, keywords, description, corpus, label from p2p_corpus where  class= 'testset'", "SQL querys testset")

tf.flags.DEFINE_string("params_dir", "./params", "Directory containing params.json")
tf.flags.DEFINE_string("PAD_WORD", "</s>", "used for pad sentence")

FLAGS = flags.FLAGS
tf.logging.set_verbosity(tf.logging.INFO)


def parse_fn(line, vocab_table, train_mode=True):
    
    def pad_fn(record, max_len):
        text = record.decode('utf-8')
        tokens = text.split()
        n=len(tokens)
        if n < max_len:
            tokens +=[FLAGS.PAD_WORD]*(max_len-n)
        if n > max_len:
            tokens = tokens[:max_len]
        tokens = [item.encode('utf-8') for item in tokens]
        return [tokens]
    
    text = tf.py_func(pad_fn, [line['text'],tf.constant(FLAGS.body_max_len)], (tf.string))
    meta = tf.py_func(pad_fn, [line['meta'],tf.constant(FLAGS.meta_max_len)], (tf.string))
   
    text.set_shape([FLAGS.body_max_len])
    meta.set_shape([FLAGS.meta_max_len])
    
    text_ids = vocab_table.lookup(text)
    meta_ids = vocab_table.lookup(meta)
    
    if train_mode:
        labels = line['labels']
        labels.set_shape([])
    
        return {'text':text_ids,'meta':meta_ids}, labels
    
    return {'text':text_ids,'meta':meta_ids}


def input_fn(sql, path_vocab,num_epochs, num_oov_buckets, shuffle, batch_size):
    
    tag_words, meta_words, labels = fetch_db(sql)
    
    vocab_table = tf.contrib.lookup.index_table_from_file(FLAGS.path_vocab,
                                                          num_oov_buckets=0,
                                                          default_value=0) 
            
    dataset = tf.data.Dataset.from_tensor_slices({'text':tag_words,
                                                  'meta':meta_words,
                                                  'labels':labels})
    
    dataset =dataset.map(lambda line: parse_fn(line,vocab_table),
                         num_parallel_calls = multiprocessing.cpu_count())
#    print("dataset.output_types",dataset.output_types)
#    
#    iterator = dataset.make_initializable_iterator()
#    next_element = iterator.get_next()
#    init_op = iterator.initializer
#    
#    with tf.Session() as sess:
#      print(sess.run(init_op))
##      print(sess.run(vocab_table))
#      tf.tables_initializer().run()
#      print("batched data :",sess.run(next_element))
   
    if shuffle:
        dataset = dataset.shuffle(buffer_size = batch_size * 10)
    dataset = dataset.repeat(num_epochs)
    dataset = dataset.batch(batch_size).prefetch(1)
    
    return dataset


def read_and_process(data):
     
    text = tf.string_split(data['text'], ' ', True)
    meta = tf.string_split(data['meta'], ' ', True)
    
    dense_text = tf.sparse_tensor_to_dense(text, default_value=FLAGS.PAD_WORD)
    dense_meta = tf.sparse_tensor_to_dense(meta, default_value=FLAGS.PAD_WORD)
    
    vocab_table = tf.contrib.lookup.index_table_from_file(FLAGS.path_vocab,
                                                      num_oov_buckets=0,
                                                      default_value=0)
    
    text_ids = vocab_table.lookup(dense_text)
    meta_ids = vocab_table.lookup(dense_meta)
    
    padding_text = tf.constant([[0, 0], [0, FLAGS.body_max_len]])
    padding_meta = tf.constant([[0, 0], [0, FLAGS.meta_max_len]])
    
    text_ids_padded = tf.pad(text_ids, padding_text)
    meta_ids_padded = tf.pad(meta_ids, padding_meta)
    
    text_id_vector = tf.slice(text_ids_padded, [0, 0], [-1, FLAGS.body_max_len])
    meta_id_vector = tf.slice(meta_ids_padded, [0, 0], [-1, FLAGS.meta_max_len])
        
    return {'text':text_id_vector,'meta':meta_id_vector}


def serving_fn():
    '''Serving input_fn that builds features from placeholders
    
    Returns 
    -------
    tf.estimator.export.ServingInputReceiver
    '''
    receiver_tensors = {
        'text': tf.placeholder(dtype=tf.string, shape=None, name='text'),
        'meta': tf.placeholder(dtype=tf.string, shape=None, name='meta')
       }
    
    data = read_and_process(receiver_tensors)
                  
    features = {
		key: tensor
                for key, tensor in data.items()
    }
  
    return tf.estimator.export.ServingInputReceiver(features, receiver_tensors)


def my_model_fn(
        features,  # This is batch_feature from input_fn 
        labels,    # This is batch_labels from input_fn
        mode,     # An instance of tf.estimator.ModeKeys
        params):   # Additional configuration 
    
    text = features['text']
    meta = features['meta']
    labels = labels
             
    embedding_placeholder = tf.placeholder(tf.float32, 
                                           [params['vocab_size'], 
                                            FLAGS.embedding_dim])
    embeddings = tf.Variable(embedding_placeholder)
    
    # shape:(batch, sentence_len, embedding_size)
    text = tf.nn.embedding_lookup(embeddings, text)
    meta = tf.nn.embedding_lookup(embeddings, meta)
    
    # add a channel dim, required by the conv2d and max_pooling2d method
    text = tf.expand_dims(text, -1) # shape:(batch, sentence_len/height, embedding_size/width, channels=1)
      
    pooled_outputs = []
    for filter_size in params["filter_sizes"]:
        conv = tf.layers.conv2d(
                text,
                filters= params["num_filters"],
                kernel_size=[filter_size, FLAGS.embedding_dim],
                strides=(1, 1),
                padding="VALID",
                activation=tf.nn.relu)
        pool = tf.layers.max_pooling2d(
                conv,
                pool_size=[FLAGS.body_max_len - filter_size + 1, 1],
                strides=(1, 1),
                padding="VALID")
        pooled_outputs.append(pool)
            
    h_pool = tf.concat(pooled_outputs, 3) # shape: (batch, 1, len(filter_size) * embedding_size, 1)
    h_pool_flat = tf.reshape(h_pool, [-1, params["num_filters"] * len(params["filter_sizes"])]) # shape: (batch, len(filter_size) * embedding_size)
    
    
    # create an LSTM cell of size 100
   # lstm_cell = tf.nn.rnn_cell.BasicLSTMCell(params['num_hidden']) 
    lstm_cell = tf.nn.rnn_cell.LSTMCell(params['num_hidden'], name ='basic_lstm_cell' )

    # create the complete LSTM
    _, final_states = tf.nn.dynamic_rnn(
        lstm_cell, meta, dtype=tf.float32)
    
    # get the final hidden states of dimensionality [batch_size x sentence_size]
    outputs = final_states.h
    
    ##concat CNN and LSTM
    concat = tf.concat([outputs,h_pool_flat], 
                        axis =1, 
                        name='concat')
    
    ##dropout layer
    dense = tf.layers.dense(inputs=concat, units=128, activation=tf.nn.relu) # 经验参数隐藏层个数128
    dropout = tf.layers.dropout(
      inputs=dense, rate=params['dropout_keep_prob'], 
      training=mode == tf.estimator.ModeKeys.TRAIN)
    
    #logits layer
    l2_regularizer = tf.contrib.layers.l2_regularizer(scale= params['l2_reg_lambda'],scope='l2_regularizer')
    logits = tf.layers.dense(inputs=dropout, 
                             units=FLAGS.num_classes,
                             kernel_regularizer=l2_regularizer,
                             activation=None)
    
    predictions = {
      # Generate predictions (for PREDICT and EVAL mode)
     "classes": tf.argmax(input=logits, axis=1),
      # Add `softmax_tensor` to the graph. It is used for PREDICT and by the
      # `logging_hook`.
      "probabilities": tf.nn.softmax(logits, name="softmax_tensor")}
    
    scaffold = tf.train.Scaffold(init_feed_dict = 
                                 {embedding_placeholder:params['embedding_matrix']})

    if mode == tf.estimator.ModeKeys.PREDICT:
        return tf.estimator.EstimatorSpec(mode=mode,
                                          predictions=predictions)
    
    # Calculate Loss (for both TRAIN and EVAL modes)
    loss = tf.losses.sparse_softmax_cross_entropy(labels=labels, logits=logits)
#    loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=logits,labels=labels))


      # Configure the Training Op (for TRAIN mode)
    if mode == tf.estimator.ModeKeys.TRAIN:
        
        optimizer = tf.train.AdamOptimizer(params['learning_rate'])
        train_op = optimizer.minimize(
                loss=loss,
                global_step=tf.train.get_global_step())
        return tf.estimator.EstimatorSpec(mode=mode,
                                          loss=loss,
                                          train_op=train_op,
                                          scaffold=scaffold)
   
  # Add evaluation metrics (for EVAL mode)
    eval_metric_ops = {
            "accuracy": tf.metrics.accuracy(labels=labels, 
                                            predictions=predictions["classes"]),
            "precision": tf.metrics.precision(labels=labels, 
                                            predictions=predictions["classes"]),
            "recall": tf.metrics.recall(labels=labels, 
                                            predictions=predictions["classes"])}
            
    return tf.estimator.EstimatorSpec(mode=mode, 
                                      loss=loss, 
                                      eval_metric_ops=eval_metric_ops,
                                      scaffold=scaffold)


def main(unused_argv):       
    # Load the parameters from the experiment params.json file in params_dir
    json_path = './params/Parallel_CNN_LSTM.json'
    assert os.path.isfile(json_path), "No json configuration file found at {}".format(json_path)
    params =utils.Params(json_path)
    
    model = gensim.models.KeyedVectors.load_word2vec_format(FLAGS.pre_trained_Word2vector, binary= False)
    vocab_list = model.index2word
    vocab_size = len(vocab_list)
    print('vocab_size is {}'.format(vocab_size))
    
    embedding_tmp = []
    for vocab in vocab_list:
        embedding_tmp.append(model.get_vector(vocab))
    embedding_matrix = np.asarray(embedding_tmp)
    
    classifier = tf.estimator.Estimator(
            model_fn = my_model_fn,
            params = {'vocab_size' : vocab_size,
                      'filter_sizes' : list(map(int, params.filter_sizes.split(","))),
                      'embedding_size': FLAGS.embedding_dim,
                      'num_filters': params.num_filters,
                      'learning_rate': params.learning_rate,
                      'dropout_keep_prob':params.dropout_keep_prob,
                      'l2_reg_lambda':params.l2_reg_lambda,
                      'num_hidden': params.num_hidden,
                      'embedding_matrix':embedding_matrix},                      
            config = tf.estimator.RunConfig(model_dir=FLAGS.model_dir,
                                            save_checkpoints_steps=params.checkpoint_every,
                                            keep_checkpoint_max=params.num_checkpoints)
   #         warm_start_from=tf.train.latest_checkpoint(
   #                  checkpoint_dir=FLAGS.model_dir)
    )
    
    early_stop = tf.contrib.estimator.stop_if_no_decrease_hook(estimator = classifier, 
                                                         metric_name="loss",
                                                         max_steps_without_decrease=10000)   
    train_spec = tf.estimator.TrainSpec(
            input_fn=lambda: input_fn(sql = FLAGS.sql_train,
				      path_vocab=FLAGS.path_vocab, 
                                      num_epochs = params.num_epochs,
                                      num_oov_buckets = FLAGS.num_oov_buckets, 
                                      shuffle=True, 
                                      batch_size = params.batch_size),
                            max_steps=FLAGS.train_steps,
                            hooks = [early_stop])
    
#    exporter = tf.estimator.LatestExporter(name='serving',
#                                           serving_input_receiver_fn= serving_fn,
#                                           exports_to_keep = 5)
    
    input_fn_for_eval = lambda: input_fn(sql = FLAGS.sql_test,
                                         path_vocab=FLAGS.path_vocab,
                                         num_epochs = 1, 
                                         num_oov_buckets = FLAGS.num_oov_buckets,
                                         shuffle=False, 
                                         batch_size = params.batch_size)
                                      
    eval_spec = tf.estimator.EvalSpec(input_fn=input_fn_for_eval, 
#                                      exporters =[exporter],
                                      throttle_secs=300)
    
    tf.estimator.train_and_evaluate(classifier, train_spec, eval_spec)
    
    classifier.export_savedmodel(export_dir_base='serving', 
                                serving_input_receiver_fn=serving_fn)
    

    
    
if __name__ == '__main__':
    tf.app.run()
