# Web_classification

tensorflow model to classify web page

The project is built to identify the P2P websites.  TextCNN and LSTM are used to extract body and meta features, respectively, then the extracted features are feed into a one-layer fully connected layer. In addition, pre-trained word2vec embedding is from [Tencent_AI_LAB](https://ai.tencent.com/ailab/nlp/data/Tencent_AILab_ChineseEmbedding.tar.gz) .


## Instal depencies

 install packages in the easiest way ``pip``

 ```  pip install -r requirments.txt ```

## Instructions

**1、Download Pre-trained embedding and unzip to folder ``embed``**

``` wget https://ai.tencent.com/ailab/nlp/data/Tencent_AILab_ChineseEmbedding.tar.gz```


**2、Run ``build_vocab.py ``,  exporting ``nwords.csv`` and ``vocab.csv`` to ``data`` folder. The former file shows the number of vocabulary of pre-trained embedding, and vocabulary are written into ``vocab.csv`` file.**

```python build_vocab.py```


**3、Modifly the parameters file**

```vim params/Parallel_CNN_LSTM.json```


**4、Train the model, the training data are import from mysql, which denfine in**

``tf.flags.DEFINE_string("sql_train", "select title, keywords, description, corpus, label from p2p_corpus where  class= 'trainset'", "SQL querys trainset")``

``tf.flags.DEFINE_string("sql_test", " select title, keywords, description, corpus, label from p2p_corpus where  class= 'testset'", "SQL querys testset")``


meta features: ``title, keywords, description ``
body feature: ``top tf-idf words in corpus``
label: ``binary label``

then train the model, the results will be export to ``output`` file:

```python custom_estimator.py```


**5、Check the logging file**

``accuracy = 0.9973114, global_step = 3801, loss = 0.0095232185, precision = 0.9769821, recall = 0.9794872``


**6、Predict the sample**

``python inference.py``


**7、Run service**

```python p2p_oss``` - which is the online service script  

```python p2p_ensemble_oss``` - which uses two classifiers to improve the precision


## Model structure

The model run TextCNN and LSTM parallelly, and combine the results, feed into the fully connected layer

![avatar](/model.png)






