# NER


## 数据来源
MRSA微软亚洲研究院的标注语料 [B,  I,  O] * [PER,  LOC,  ORG]

## 特征选取与定义
### 字符角色定义
BIESO (其他的字符角色定义还有很多)
- B, 即Begin, 表示开始
- I, 即Intermediate, 表示中间
- E, 即End, 表示结尾
- S, 即Single, 表示单个字符
- O, 即Other, 表示其他, 用于标记无关字符

### 实体类别定义
命名实体识别中常见的实体类别有：
- PERSON 人名
- LOCATION 地名
- ORGNIZATION 机构名
- TIME 时间

但是在其他领域, 如医学领域, 实体类别可细分为：
- TREATMENT 治疗方式
- BODY 身体部位
- SIGNS 疾病症状
- CHECK 医学检查
- DISEASE 疾病实体
- MEDICINE 药物实体

在训练预料中, 结合字符角色定义和实体类别定义, 一个句子（一个句子即为一个序列, 以换行符'\n'分隔）按照字符（word/char）可标注为：
```
美 B-LOC
国 I-LOC
的 O
华 B-PER
莱 B-PER
士 B-PER
,  O
我 O
和 O
他 O
谈 O
笑 O
风 O
生 O
。 O

```
针对语料的不同, 亦可在此基础上添加其他特征, 如增加“非汉字字符串”的定义。


## 模型选取
NER通常被转换成文本表示和分类任务, 类别即我们根据特征定义的**字符角色和实体类别的排列组合**

### CRF
条件随机场, 传统的机器学习模型，在深度学习模型中通常作为最后的标签推断层。

### BiLSTM-CRF

BiLSTM是Bi-directional Long Short-Term Memory的缩写, 是由前向LSTM与后向LSTM组合而成, 常被用来建模上下文信息。

1. 使用预训练字向量作为embedding层输入, 否则, 随机化初始字向量；
2. 然后经过两个双向LSTM层进行编码, 编码后加入dense全连接层；
3. 最后输入CRF层进行序列标注。

在这里, 我们使用开源的基于中文维基百科训练的100维的字向量作为embedding层（实验证明随机初始化字向量，在同领域的验证集也能达到较好的效果）， 在`blstm_ner`使用`tag2label`定义标签类别, 视具体任务而定。

在训练BiLSTM_CRF模型的过程中发现, 设置`learning_rate=0.001`, `batch_size=64`的收敛效率较好, 在第十几个epoch就达到了较好的效果。

learning_rate一般通过指数衰减调参，学习率决定了参数每次更新的幅度, 如果幅度过大, 那么可能导致参数在极优值得两侧来回移动；如果幅度过小, 虽然能保证收敛性, 但这会大大降低优化速度。

batch_size一般从128开始上下浮动。一个batch即批训练一次神经网络, 计算损失函数, 利用梯度下降更新网络参数。batch越大其越能代表整个数据集的分布, 但越大也意味着计算量越大, 收敛越慢。

注意，由于LSTM的遗忘机制，在训练时dropout要设置得大一点，一般取0.5。


### BERT
[BERT](https://github.com/google-research/bert)（Bidirectional Encoder Representations from Transfoemers）, 是预训练好的语言模型, 可应用于不同的NLP下游任务。

扩展阅读： [从Word Embedding到Bert模型—自然语言处理中的预训练技术发展史](https://zhuanlan.zhihu.com/p/49271699)

在这里, 我们使用预训练的Bert中文模型作为embedding层, 在`get_lables`里定义标签类别, 视具体任务而定。
```python
return ["X",  "B-LOC",  "I-LOC",  "B-PER",  "I-PER",  "B-ORG",  "I-ORG",  "O",  "[CLS]",  "[SEP]"]
```

在训练Bert模型的过程中, 使用的是开源模型的参数`learning_rate=2e-5`, `batch_size=32`, 模型在验证集上的效果已经很好, 所以没有再调参数。


## 实验

### Requirements

- Python 3.5 (recommended)
- TensorFlow >= 1.10.0
- numpy >= 1.14.3
- Anaconda (recommended on Windows)

### 注意
- BERT在CPU上运行可能会卡机, 模型训练耗时约day+
- [Google Colab](https://colab.research.google.com) 提供了免费的GPU, 需要科学上网
- 由于Git托管体积大的文件不方便，相关的训练集、词向量、预训练模型、下游任务训练模型已上传到网盘

[链接](https://pan.baidu.com/s/1zVR7paWVE7WL4L87BDV6rA)  提取码: h8st 

### 模型训练
cmd.exe不支持sh脚本, 可在git bash上执行, 或在控制台带参数运行, 对于不同的语料、输出和参数, 需要相应地修改sh脚本

1. BERT / BERT + CRF / BERT + BiLSTM / BERT + BiLSTM + CRF
```bash
bash run_bert_ner.sh
```


```shell
#!/usr/bin/env bash

# 部分参数在bert_ber中有默认值
python bert_ner.py
    --do_lower_case=False \
    --do_train=True   \
    --do_eval=True   \
    --do_test=True \
    --data_dir=./data/MSRA/   \
    --vocab_file=data/vocab.txt   \
    --bert_config_file=chinese_L-12_H-768_A-12/bert_config.json   \
    --init_checkpoint=chinese_L-12_H-768_A-12/bert_model.ckpt   \
    --max_seq_length=128   \
    --train_batch_size=32   \
    --learning_rate=2e-5   \
    --num_train_epochs=10.0   \
    --dropout_rate=0.5   \
    --output_dir=./output/MSRA/bert_bilstm_crf/   \
    --bilstm=True   \
    --crf=True
```

由于训练样本大，只需训练4-5个epoch就能达到不错的效果，再多有点过拟合。

2. BiLSTM / BiLSTM + CRF
```bash
bash run_bilstm_ner.sh
```

```shell
python blstm_ner.py\
    --data_dir=./data/MSRA   \
    --output_dir=./output/MSRA/char2vec_blstm_crf/   \
    --vocab_file=./data/wiki/char2id.pkl   \
    --embedding_file=./data/wiki/char2vec.txt   \
    --embedding_source=./data/wiki/wiki_100.utf8.txt   \
    --lr=0.001   \
    --num_train_epochs=40   \
    --batch_size=64   \
    --CRF=True   \
    --mode=train
```


## 应用

### 输入新文本, 抽取目标实体

若从文件读入文本，设置`file_input=<file_name under data_dir>`，默认在`txt2seq.py`中以`r'([。？！])'`切割句子，如果切割后序列长度仍超出128-2(CLS和SEP)，则继续以`；`或`，`切割。

若从控制台交互输入文本，设置`raw_input=True`，长度超过128-2会自动截断（尝试训练max_seq_length=256的模型但是提示内存溢出）。

在预测过程中，可能会出现字典文件`vocab.txt`未包含的字符，以`[UNK]`输出，可手动更新`[unused1~99]`另存为`vocab_update`文件。

目标实体以句子为单位输出。

另外，可在`seq2entity.py`中加入规则，抽取其他类型的实体，如《》。

(Attention: args 不支持 raw_input=False，只要加上该参数就会设置为True，所以还是flags更灵活一点吧)

1. BERT / BERT + CRF / BERT + BiLSTM / BERT + BiLSTM + CRF
```bash
cd application
bash run_predict.sh
```

`data_dir`不是训练语料的目录，而是应用文本的目录，最好分开，最终输出的实体文件和日志文件也会保存在这里.

`init_checkpoint`是训练好的模型，预测的时候会从该文件里重加载神经网络模型的参数.

`output_dir`是训练模型的保存路径，用来重载某些依赖文件，如`label2id.pkl`.


2. BiLSTM / BiLSTM + CRF
```shell
python blstm_ner.py\
    --data_dir=./data/ywsz/   \
    --output_dir=./output/ChinaDaily/blstm_crf/   \
    --vocab_file=./data/wiki/char2id.pkl   \
    --embedding_file=./data/wiki/char2vec.txt   \
    --CRF=True   \
    --mode=predict   \
```

- [ ] incremental learning 






