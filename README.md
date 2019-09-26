# ChineseNER


## 数据来源
目前常用的、成熟的NER语料有：
- 人民日报的语料 [B, I, O] * [PER, LOC, ORG]
- MRSA微软亚洲研究院的语料 [B, I, O] * [PER, LOC, ORG]

这些都是通用领域的现代语料。

另外，可以去一些会议和相关竞赛找专业领域较强的语料，如CCKS全国知识图谱与语义计算大会

## 特征选取与定义
### 字符角色定义
BIESO (其他的字符角色定义还有很多)
- B，即Begin，表示开始
- I，即Intermediate，表示中间
- E，即End，表示结尾
- S，即Single，表示单个字符
- O，即Other，表示其他，用于标记无关字符

### 实体类别定义
命名实体识别中常见的实体类别有：
- PERSON 人名
- LOCATION 地名
- ORGNIZATION 机构名
- TIME 时间

但是在其他领域，如医学领域，实体类别可细分为：
- TREATMENT 治疗方式
- BODY 身体部位
- SIGNS 疾病症状
- CHECK 医学检查
- DISEASE 疾病实体
- MEDICINE 药物实体

所以在训练预料中，结合字符角色定义和实体类别定义，一个句子（一个句子即为一个序列，以换行符'\n'分隔）按照字符（word/char）可标注为：
```
美 B-LOC
国 I-LOC
的 O
华 B-PER
莱 B-PER
士 B-PER
， O
我 O
和 O
他 O
谈 O
笑 O
风 O
生 O
。 O

```
针对语料的不同，亦可在此基础上添加其他特征，如增加“非汉字字符串”的定义，



## 模型选取
NER通常被转换成分类任务，类别即我们根据特征定义的**字符角色和实体类别的排列组合**

## CRF
条件随机场，通常作为最后一层的标签推断层。

### Bi-LSTM-CRF
1. 使用预训练字向量作为embedding层输入，否则，随机化初始字向量；
2. 然后经过两个双向LSTM层进行编码，编码后加入dense全连接层；
3. 最后送入CRF层进行序列标注。


### BERT
[BERT](https://github.com/google-research/bert)（Bidirectional Encoder Representations from Transfoemers），是预训练好的语言模型，可应用于不同的NLP下游任务。

扩展阅读： [从Word Embedding到Bert模型—自然语言处理中的预训练技术发展史](https://zhuanlan.zhihu.com/p/49271699)


BERT的特点在于：
1. 预训练
预训练模型的开销很大，Google提供了基于百科类语料的中文模型，另外有论文提出了基于高质量英文论文的[SCIBERT](https://github.com/allenai/scibert)

2. 双向编码

3. 语言遮罩

>[X]  Mask掩盖，能更有效学习

>[CLS]  每个序列的第一个 token 始终是特殊分类嵌入（special classification embedding），对应于该 token 的最终隐藏状态（即Transformer的输出）被用于分类任务的聚合序列表示。如果没有分类任务的话，这个向量是被忽略的。

>[SEP]  用于分隔一对句子的特殊符号。有两种方法用于分隔句子：第一种是使用特殊符号 SEP；第二种是添加学习句子 A 嵌入到第一个句子的每个 token 中，句子 B 嵌入到第二个句子的每个 token 中，即[CLS]+ [sent A]+ [SEP] + [sent B] + [SEP]。如果是单个输入的话，就只使用句子 A ，即[CLS]+ [sent A]+ [SEP]。



在这里，我们使用预训练的Bert中文模型作为embedding层，考虑到NER任务的特殊性，需要对一般的分类模型进行改造：

1. 在`run_classifier.py`示例中的任务都是针对**句子对输入**（next sentence prediction,），而在NER任务中只有单个句子输入

2. 在`get_lables`里定义标签类别，视具体任务而定
```python
#return ["O", "B-PER", "I-PER", "B-ORG", "I-ORG", "B-LOC", "I-LOC", "X", "[CLS]", "[SEP]"]
return ["X", "B", "I", "E", "S", "O", "[CLS]", "[SEP]"]
```


## 实验

### Requirements

- Python 3.5 (recommended)
- TensorFlow >= 1.10.0
- numpy >= 1.14.3
- Anaconda (recommended on Windows)

### Tips
- 在CPU上运行可能会卡机，建议在实验室电脑上训练
- 模型训练耗时约day+


### How to run
cmd.exe不支持sh脚本，可在git bash上执行，或在控制台带参数运行`python bert_ner.py`
```bash
bash run_bert_ner.sh
```
对于不同的语料、输出和参数，需要相应地修改`run_ner.sh`
```shell
#!/usr/bin/env bash

python bert_ner.py\
    --task_name="NER"  \
    --do_lower_case=False \
    --do_train=True   \
    --do_eval=True   \
    --do_predict=True \
    --data_dir=[data_dir]   \
    --vocab_file=chinese_L-12_H-768_A-12/vocab.txt  \
    --bert_config_file=chinese_L-12_H-768_A-12/bert_config.json \
    --init_checkpoint=chinese_L-12_H-768_A-12/bert_model.ckpt   \
    --max_seq_length=128   \
    --train_batch_size=32   \
    --learning_rate=2e-5   \
    --num_train_epochs=4.0   \
    --output_dir=[save path of model]
    --bilstm=[whether to add bilstm_crf_layer, default to True]
    --crf_only=[whether to use crf_layer only, default to False]


# 以人民日报语料为例
# 其他参数在bert_ber中有默认值
python bert_ner.py
    --data_dir=./data/ChinaDaily/
    --bert_config_file=chinese_L-12_H-768_A-12/bert_config.json
    --init_checkpoint=chinese_L-12_H-768_A-12/bert_model.ckpt
    --vocab_file=data/vocab.txt
    --output_dir=./output/bert/ChinaDaily
```


## 应用TODO

- [ ] 根据预测输出的标签，抽取目标实体；
- [ ] OOV(Out of vocabulary)的问题，有实验证明BERT的效果较好；
- [ ] transfer learning 迁移学习 ；






