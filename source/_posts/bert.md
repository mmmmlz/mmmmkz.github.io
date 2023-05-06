---
title: DC竞赛复盘
date: 2022-1-21 22:05:53
categories:
- Notes
- NLP
tags:
- Rec System
mathjax: true
---

秋招拿到了推荐相关的office，和未来老板沟通几次后也产生极大的兴趣，感觉以后要放弃Cv啦（快被卷死了~）。由于自己之前没做过推荐相关，害怕入职后会跟不上节奏（听说部门节奏很快。。） 最近正好没事，学习的同时找了个推荐相关的比赛练练手。<!-- more -->

这是一个分类任务的比赛，主要要根据用户的个人信息，行为信息和订单信息来预测用户的下一个订单是否符合特定要求。由于最近正好看到了i2i召回相关的内容，于是尝试用了下word2vec 来生成embedding。我使用的方法是仅利用用户的行为信息，主要的思路是：**将每个动作通过 word2vec 转化为 embedding 表示，然后将动作序列转化为 embedding 序列并作为 CNN/RNN 的输入。** 下面依次介绍通过 word2vec 获得动作 embedding，将 embedding 作为CNN的输入和将embedding作为RNN的输入这三部分内容。哎，还是做什么首先想到用CNN。。

## word2vec 获取动作 embedding

word2vec 是一个很著名的无监督算法了，这个算法最初在NLP领域提出，可以通过词语间的关系构建词向量，进而通过词向量可获取词语的语义信息，如词语意思相近度等。而将 word2vec 应用到动作序列中，主要是受到了知乎上[这个答案](https://www.zhihu.com/question/25269336/answer/49188284)的启发。因为 word2vec 能够挖掘序列中各个元素之间的关系信息，这里如果将每个动作看成是一个单词，然后通过 word2vec 得出每个动作的 embedding 表示，那么这些 embedding 之间会存在一定的关联程度，再将动作序列转为 embedding 序列，作为 CNN 或 RNN 的输入便可挖掘整个序列的信息。

这里训练动作 embedding 的方法跟训练 word embedding 的方法一致，将每个户的每个动作看做一个单词、动作序列看做一篇文章即可。训练时采用的是 `gensim`, 训练的代码很简单，embedding 的维度设为 300, `filter_texts`中每一行是一各用户的行为序列，行为之间用空格隔开。

```python
from gensim.models import word2vec
vector_length = 300
model = word2vec.Word2Vec(filter_texts, size = vector_length, window=2, workers=4)

```

由于动作类型只有9种（1~9），也就是共有 9 个不同的单词，因此可将这 9 个动作的 embedding 存在一个 `np.ndarray` 中，然后作为后面 CNN/RNN 前的 embedding layer 的初始化权重。注意这里还添加了一个动作 0 ，原因是 CNN 的输入要求长度一致，因此对于长度达不到要求长度的序列，需要在前面补 0（补其他的不是已知的动作也可以）。代码如下

```python
import numpy as np
embedding_matrix = np.zeros((10, vector_length))
for i in range(1, 10):
    embedding_matrix[i] = model.wv[str(i)]
```

## CNN 对动作序列建模

CNN 采用的模型就是普通的两层卷积，一开始没想着CNN能有多大提升，搭建方式就看代码吧。

```python
NUM_EPOCHS = 100
BATCH_SIZE = 64
DROP_PORB = (0.5, 0.8)
NUM_FILTERS = (64, 32)
FILTER_SIZES = (2, 3, 5, 8)
HIDDEN_DIMS = 1024
FEATURE_DIMS = 256
ACTIVE_FUNC = 'relu'

sequence_input = Input(shape=(max_len, ), dtype='int32')
embedded_seq = embedding_layer(sequence_input)

# Convolutional block
conv_blocks = []
for size in FILTER_SIZES:
    conv = Convolution1D(filters=NUM_FILTERS[0],
                         kernel_size=size,
                         padding="valid",
                         activation=ACTIVE_FUNC,
                         strides=1)(embedded_seq)
    conv = Convolution1D(filters=NUM_FILTERS[1],
                         kernel_size=2,
                         padding="valid",
                         activation=ACTIVE_FUNC,
                         strides=1)(conv)
    conv = Flatten()(conv)
    conv_blocks.append(conv)

model_tmp = Concatenate()(conv_blocks) if len(conv_blocks) > 1 else conv_blocks[0]
model_tmp = Dropout(DROP_PORB[1])(model_tmp)
model_tmp = Dense(HIDDEN_DIMS, activation=ACTIVE_FUNC)(model_tmp)
model_tmp = Dropout(DROP_PORB[0])(model_tmp)
model_tmp = Dense(FEATURE_DIMS, activation=ACTIVE_FUNC)(model_tmp)
model_tmp = Dropout(DROP_PORB[0])(model_tmp)
model_output = Dense(1, activation="sigmoid")(model_tmp)
model = Model(sequence_input, model_output)

opti = optimizers.SGD(lr = 0.01, momentum=0.8, decay=0.0001)

model.compile(loss='binary_crossentropy',
              optimizer = opti,
              metrics=['binary_accuracy'])

model.fit(x_tra, y_tra, batch_size = BATCH_SIZE, validation_data = (x_val, y_val))
```

由于最后要求的是 auc 指标，但是 Keras 中并没有提供，而 accuracy 与 auc 还是存在一定差距的，因此可以在每个epoch后通过 sklearn 计算auc，具体代码如下

```python
from sklearn import metrics
for i in range(NUM_EPOCHS):
    model.fit(x_tra, y_tra, batch_size = BATCH_SIZE, validation_data = (x_val, y_val))
    y_pred = model.predict(x_val)
    val_auc = metrics.roc_auc_score(y_val, y_pred)
    print('val_auc:{0:5f}'.format(val_auc))
```

这种方法最终的准确率约为 0.86，auc 约为0.84。个人感觉还是不错了，名次在50左右- -！不过感觉在调下参应该会有一些提升。
