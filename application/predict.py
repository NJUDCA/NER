#! usr/bin/env python3
# -*- coding:utf-8 -*-
"""
Copyright 2018 The Google AI Language Team Authors.
BASED ON Google_BERT.
@Author:xiongxin
Adjust code for BiLSTM plus CRF based on zhoukaiyin.
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import collections
import os
import sys
sys.path.append("..")
from bert import modeling
from bert import tokenization
from blstm_crf.bilstm_crf_layer import BiLSTM_CRF
from application.txt2seq import Txt2Seq
from application.seq2entity import Seq2Entity
import tensorflow as tf
import pickle
import logging
import numpy as np

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
'''
0 = all messages are logged (default behavior)
1 = INFO messages are not printed
2 = INFO and WARNING messages are not printed
3 = INFO, WARNING, and ERROR messages are not printed
'''
flags = tf.flags
FLAGS = flags.FLAGS


flags.DEFINE_string("data_dir", None,
                    "The input dir.")

flags.DEFINE_string("bert_config_file", None,
                    "The config json file corresponding to the pre-trained BERT model.")

flags.DEFINE_string("task_name", "NER",
                    "The name of the task to train.")

flags.DEFINE_string("output_dir", None,
                    "The output directory where the model checkpoints will be written.")

# Other parameters
flags.DEFINE_string("init_checkpoint", None,
                    "Initial checkpoint (usually from a pre-trained BERT model).")

flags.DEFINE_bool("do_lower_case", True,
                  "Whether to lower case the input text. Should be True for uncased.")

flags.DEFINE_integer("max_seq_length", 128,
                     "The maximum total input sequence length after WordPiece tokenization.")

flags.DEFINE_bool("do_predict", True,
                  "Whether to run the model in inference mode on the test set.")

flags.DEFINE_integer("train_batch_size", 32,
                     "Total batch size for training.")

flags.DEFINE_integer("eval_batch_size", 8,
                     "Total batch size for eval.")

flags.DEFINE_integer("predict_batch_size", 8,
                     "Total batch size for predict.")

flags.DEFINE_float("learning_rate", 5e-5,
                   "The initial learning rate for Adam.")

flags.DEFINE_float("dropout_rate", 0.1,
                   "The initial learning rate for Adam.")

flags.DEFINE_integer("save_checkpoints_steps", 1000,
                     "How often to save the model checkpoint.")

flags.DEFINE_integer("iterations_per_loop", 1000,
                     "How many steps to make in each estimator call.")

flags.DEFINE_string("vocab_file", None,
                    "The vocabulary file that the BERT model was trained on.")

flags.DEFINE_bool("use_tpu", False,
                  "Whether to use TPU or GPU/CPU.")

tf.flags.DEFINE_string(
    "tpu_name", None,
    "The Cloud TPU to use for training. This should be either the name "
    "used when creating the Cloud TPU, or a grpc://ip.address.of.tpu:8470 "
    "url.")

tf.flags.DEFINE_string(
    "tpu_zone", None,
    "[Optional] GCE zone where the Cloud TPU is located in. If not "
    "specified, we will attempt to automatically detect the GCE project from "
    "metadata.")

tf.flags.DEFINE_string(
    "gcp_project", None,
    "[Optional] Project name for the Cloud TPU-enabled project. If not "
    "specified, we will attempt to automatically detect the GCE project from "
    "metadata.")

tf.flags.DEFINE_string("master", None,
                       "[Optional] TensorFlow master URL.")

flags.DEFINE_integer("num_tpu_cores", 8,
                     "Only used if `use_tpu` is True. Total number of TPU cores to use.")

flags.DEFINE_bool("bilstm", True,
                  "use bilstm + crf.")

flags.DEFINE_bool("crf_only", False,
                  "use crf only.")

# lstm params
flags.DEFINE_integer('lstm_size', 128,
                     'size of lstm units')

flags.DEFINE_integer('num_layers', 1,
                     'number of rnn layers, default is 1')

flags.DEFINE_string('cell', 'lstm',
                    'which rnn cell used')

flags.DEFINE_bool('raw_input', True,
                  'default to input text for prediction')


class InputExample(object):
    """
    A single training/test example for simple sequence classification.
    """

    def __init__(self, guid, text, label=None):
        """
        construct a input example
        :param guid: unique id for the example
        :param text: The untokenized text of the first sequence. For single
            sequence tasks, only this sequence must be specified.
        :param label: (Optional) string. The label of the example. This should be
            specified for train and dev examples, but not for test examples.
        """
        self.guid = guid
        self.text = text
        self.label = label


class InputFeatures(object):
    """
    A single set of features of data.
    """
    def __init__(self, input_ids, input_mask, segment_ids, label_ids, is_real_example=True):
        self.input_ids = input_ids
        self.input_mask = input_mask
        self.segment_ids = segment_ids
        self.label_ids = label_ids
        self.is_real_example = is_real_example


class DataProcessor(object):
    """
    Base class for data converters for sequence classification data sets.
    """

    def get_train_examples(self, data_dir):
        """
        Gets a collection of `InputExample`s for the train set.
        """
        raise NotImplementedError()

    def get_dev_examples(self, data_dir):
        """
        Gets a collection of `InputExample`s for the dev set.
        """
        raise NotImplementedError()

    def get_labels(self):
        """
        Gets the list of labels for this data set.
        """
        raise NotImplementedError()

    @classmethod
    def _read_data(cls, input_file):
        """
        customized to read train/dev/test data here!
        """
        with open(input_file, 'r', encoding='utf-8') as f:
            lines = []
            words = []
            labels = []
            for line in f:
                contents = line.strip()
                word = line.strip().split(' ')[0]
                label = line.strip().split(' ')[-1]
                if contents.startswith("-DOCSTART-"):
                    words.append('')
                    continue
                if len(contents) == 0:
                    l = ' '.join([label for label in labels if len(label) > 0])
                    w = ' '.join([word for word in words if len(word) > 0])
                    lines.append([l, w])
                    words = []
                    labels = []
                    continue
                words.append(word)
                labels.append(label)
            return lines


class NerProcessor(DataProcessor):
    def _create_example(self, lines, set_type):
        example = []
        for (i, line) in enumerate(lines):
            guid = '{}-{}'.format(set_type, i)
            text = tokenization.convert_to_unicode(line[1])
            label = tokenization.convert_to_unicode(line[0])
            example.append(InputExample(guid=guid, text=text, label=label))
        return example

    def get_train_examples(self, data_dir):
        return self._create_example(
            self._read_data(os.path.join(data_dir, 'train.txt')), 'train'
        )

    def get_dev_examples(self, data_dir):
        return self._create_example(
            self._read_data(os.path.join(data_dir, 'dev.txt')), 'dev'
        )

    def get_predict_examples(self, data_dir):
        return self._create_example(
            self._read_data(os.path.join(data_dir, "predict.txt")), "test")

    def get_labels(self):
        '''
        based on ChinaDaily corpus
        'X' is used to represent "##eer","##soo" and char not in vocab!
        '''
        return ["X", "B-LOC", "I-LOC", "B-PER", "I-PER", "B-ORG", "I-ORG", "O", "[CLS]"]


def convert_single_example(ex_index, example, label_list, max_seq_length, tokenizer, mode):
    """
    :param ex_index: example num
    :param example:
    :param label_list: all labels
    :param max_seq_length:
    :param tokenizer: WordPiece tokenization
    :param mode:
    :return: feature
    IN this part we should rebuild input sentences to the following format.
    example:[Jim,Hen,##son,was,a,puppet,##eer]
    labels: [B,I,X,O,O,O,X]
    """
    label_map = {}
    # here start with zero which means that "[PAD]" is zero
    # start with 1, 0 for paddding
    for i, label in enumerate(label_list, 1):
        label_map[label] = i
    with open(FLAGS.output_dir + '/label2id.pkl', 'wb+') as wf:
        pickle.dump(label_map, wf)

    textlist = example.text.split(' ')
    labellist = example.label.split(' ')
    tokens = []
    labels = []
    for i, (word, label) in enumerate(zip(textlist,labellist)):
        token = tokenizer.tokenize(word)
        tokens.extend(token)
        for m in range(len(token)):
            if m == 0:
                labels.append(label)
            else:
                labels.append('X')
    # only Account for [CLS] with "- 1".
    # account for ending signal [SEP], with "- 2"
    if len(tokens) >= max_seq_length - 1:
        tokens = tokens[0: max_seq_length - 1]
        labels = labels[0: max_seq_length - 1]
    ntokens = []
    segment_ids = []
    label_ids = []
    # begining signal [CLS]
    ntokens.append("[CLS]")
    segment_ids.append(0)
    label_ids.append(label_map["[CLS]"])
    for i, token in enumerate(tokens):
        ntokens.append(token)
        segment_ids.append(0)
        label_ids.append(label_map[labels[i]])
    '''
    after that we don't add "[SEP]" because we want a sentence don't have
    stop tag, because i think its not very necessary.
    or if add "[SEP]" the model even will cause problem, special the crf layer was used.
    '''
    # endinging signal [SEP]
    # ntokens.append("[SEP]")
    # segment_ids.append(0)
    # label_ids.append(label_map["[SEP]"])

    input_ids = tokenizer.convert_tokens_to_ids(ntokens)
    input_mask = [1] * len(input_ids)
    #use zero to padding sequence
    while len(input_ids) < max_seq_length:
        input_ids.append(0)
        input_mask.append(0)
        segment_ids.append(0)
        label_ids.append(0)
        ntokens.append("**NULL**")
    assert len(input_ids) == max_seq_length
    assert len(input_mask) == max_seq_length
    assert len(segment_ids) == max_seq_length
    assert len(label_ids) == max_seq_length

    if ex_index < 5:
        logging.info("*** Example ***")
        logging.info("guid: %s" % (example.guid))
        logging.info("tokens: %s" % " ".join(
            [tokenization.printable_text(x) for x in tokens]))
        logging.info("input_ids: %s" % " ".join([str(x) for x in input_ids]))
        logging.info("input_mask: %s" % " ".join([str(x) for x in input_mask]))
        logging.info("segment_ids: %s" % " ".join([str(x) for x in segment_ids]))
        logging.info("label_ids: %s" % " ".join([str(x) for x in label_ids]))
        logging.info("ntokens: %s" % " ".join([str(x) for x in ntokens]))

    feature = InputFeatures(
        input_ids=input_ids,
        input_mask=input_mask,
        segment_ids=segment_ids,
        label_ids=label_ids,
    )
    # we need ntokens because if we do predict it can help us return to original token.
    return feature, ntokens, label_ids


def file_based_convert_examples_to_features(
        examples, label_list, max_seq_length, tokenizer, output_file, mode=None):
    writer = tf.python_io.TFRecordWriter(path=output_file)
    # writer = tf.io.TFRecordWriter(path=output_file)
    batch_tokens = []
    batch_labels = []
    for ex_index, example in enumerate(examples):
        if ex_index % 5000 == 0:
            logging.info('Writing example {} of {}'.format(ex_index, len(examples)))
        feature, ntokens, label_ids = convert_single_example(ex_index, example, label_list, max_seq_length, tokenizer, mode)
        batch_tokens.extend(ntokens)
        batch_labels.extend(label_ids)

        def create_int_feature(values):
            f = tf.train.Feature(int64_list=tf.train.Int64List(value=list(values)))
            return f

        features = collections.OrderedDict()
        features["input_ids"] = create_int_feature(feature.input_ids)
        features["input_mask"] = create_int_feature(feature.input_mask)
        features["segment_ids"] = create_int_feature(feature.segment_ids)
        features["label_ids"] = create_int_feature(feature.label_ids)
        tf_example = tf.train.Example(features=tf.train.Features(feature=features))
        writer.write(tf_example.SerializeToString())
    writer.close()
    return batch_tokens, batch_labels


def file_based_input_fn_builder(input_file, seq_length, is_training, drop_remainder):
    name_to_features = {
        "input_ids": tf.FixedLenFeature([seq_length], tf.int64),
        "input_mask": tf.FixedLenFeature([seq_length], tf.int64),
        "segment_ids": tf.FixedLenFeature([seq_length], tf.int64),
        "label_ids": tf.FixedLenFeature([seq_length], tf.int64),
    }

    def _deocde_record(record, name_to_features):
        example = tf.parse_single_example(record, name_to_features)
        for name in list(example.keys()):
            t = example[name]
            if t.dtype == tf.int64:
                t = tf.to_int32(t)
            example[name] = t
        return example

    def input_fn(params):
        batch_size = params["batch_size"]
        d = tf.data.TFRecordDataset(input_file)
        if is_training:
            d = d.repeat()
            d = d.shuffle(buffer_size=100)
        d = d.apply(tf.contrib.data.map_and_batch(
            lambda record: _deocde_record(record, name_to_features),
            batch_size=batch_size,
            drop_remainder=drop_remainder
        ))
        return d

    return input_fn


def hidden2tag(hiddenlayer, numclass):
    # tf.keras.layers.Dense 封装了output = activation(tf.matmul(input, kernel) + bias)
    # 相当于全连接层的线性变换
    linear = tf.keras.layers.Dense(numclass, activation=None)
    return linear(hiddenlayer)


def softmax_layer(logits, labels, num_labels, mask):
    logits = tf.reshape(logits, [-1, num_labels])
    labels = tf.reshape(labels, [-1])
    mask = tf.cast(mask, dtype=tf.float32)
    one_hot_labels = tf.one_hot(labels, depth=num_labels, dtype=tf.float32)
    loss = tf.losses.softmax_cross_entropy(logits=logits, onehot_labels=one_hot_labels)
    loss *= tf.reshape(mask, [-1])
    loss = tf.reduce_sum(loss)
    total_size = tf.reduce_sum(mask)
    total_size += 1e-12 # to avoid division by 0 for all-0 weights
    loss /= total_size
    # predict not mask we could filtered it in the prediction part.
    probabilities = tf.math.softmax(logits, axis=-1)
    predict = tf.math.argmax(probabilities, axis=-1)
    return loss, predict


def create_model(bert_config, is_training, input_ids, input_mask,
                 segment_ids, labels, num_labels, use_one_hot_embeddings):
    """
    :param bert_config: bert 配置
    :param is_training:
    :param input_ids: 数据的idx 表示
    :param input_mask:
    :param segment_ids:
    :param labels: 标签的idx 表示
    :param num_labels: 类别数量
    :param use_one_hot_embeddings:
    :return:
    """
    # 使用数据加载BertModel,获取对应的字embedding
    model = modeling.BertModel(
        config=bert_config,
        is_training=is_training,
        input_ids=input_ids,
        input_mask=input_mask,
        token_type_ids=segment_ids,
        use_one_hot_embeddings=use_one_hot_embeddings
    )
    # [batch_size, seq_length, embedding_size]
    # use model.get_sequence_output() to get token-level output
    output_layer = model.get_sequence_output()

    if is_training:
        output_layer = tf.nn.dropout(output_layer, FLAGS.dropout_rate)

    if FLAGS.bilstm:
        '''
        used = tf.sign(tf.abs(input_ids))
        lengths = tf.reduce_sum(used, reduction_indices=1)
        '''
        # [batch_size] 大小的向量，包含了当前batch中的序列长度
        lengths = tf.reduce_sum(input_mask, axis=1)
        max_seq_length = output_layer.shape[1].value

        bilstm_crf = BiLSTM_CRF(embedded_chars=output_layer, lstm_size=FLAGS.lstm_size, cell_type=FLAGS.cell, num_layers=FLAGS.num_layers,
                                dropout_rate=FLAGS.dropout_rate, num_labels=num_labels, max_seq_length=max_seq_length, labels=labels,
                                lengths=lengths, is_training=is_training, crf_only=FLAGS.crf_only)
        loss, logits, predict = bilstm_crf.add_bilstm_crf_layer()
    else:
        logits = hidden2tag(output_layer, num_labels)
        logits = tf.reshape(logits, [-1, FLAGS.max_seq_length, num_labels])
        loss, predict = softmax_layer(logits, labels, num_labels, input_mask)

    return (loss, logits, predict)


def model_fn_builder(bert_config, num_labels, init_checkpoint, learning_rate,
                     num_train_steps, num_warmup_steps, use_tpu, use_one_hot_embeddings):
    def model_fn(features, labels, mode, params):
        logging.info("*** Features ***")
        for name in sorted(features.keys()):
            logging.info("  name = {}, shape = {}".format(name, features[name].shape))
        input_ids = features["input_ids"]
        input_mask = features["input_mask"]
        segment_ids = features["segment_ids"]
        label_ids = features["label_ids"]
        is_training = (mode == tf.estimator.ModeKeys.TRAIN)

        (total_loss, logits, pred_ids) = create_model(
            bert_config, is_training, input_ids, input_mask, segment_ids,
            label_ids, num_labels, use_one_hot_embeddings)

        vars = tf.trainable_variables()
        scaffold_fn = None
        if init_checkpoint:
            (assignment_map, initialized_variable_names) = modeling.get_assignment_map_from_checkpoint(
                vars, init_checkpoint)
            tf.train.init_from_checkpoint(init_checkpoint, assignment_map)
            if use_tpu:
                def tpu_scaffold():
                    tf.train.init_from_checkpoint(init_checkpoint, assignment_map)
                    return tf.train.Scaffold()
                scaffold_fn = tpu_scaffold
            else:
                tf.train.init_from_checkpoint(init_checkpoint, assignment_map)
        logging.info("**** Trainable Variables ****")

        for var in vars:
            init_string = ""
            if var.name in initialized_variable_names:
                init_string = ", *INIT_FROM_CKPT*"
            logging.info("  name = {}, shape = {} {}".format(var.name, var.shape, init_string))

        output_spec = None

        output_spec = tf.contrib.tpu.TPUEstimatorSpec(
            mode=mode, predictions=pred_ids, scaffold_fn=scaffold_fn)
        return output_spec

    return model_fn


def main(_):
    logging.info(str(FLAGS))

    processors = {
        "ner": NerProcessor
    }

    bert_config = modeling.BertConfig.from_json_file(FLAGS.bert_config_file)

    if FLAGS.max_seq_length > bert_config.max_position_embeddings:
        raise ValueError("Cannot use sequence length {} because the BERT model, was only trained up to sequence length {}".format(
            FLAGS.max_seq_length, bert_config.max_position_embeddings))

    task_name = FLAGS.task_name.lower()
    if task_name not in processors:
        raise ValueError("Task not found: {}".format(task_name))

    processor = processors[task_name]()

    label_list = processor.get_labels()

    tokenizer = tokenization.FullTokenizer(
        vocab_file=FLAGS.vocab_file, do_lower_case=FLAGS.do_lower_case)
    tpu_cluster_resolver = None
    if FLAGS.use_tpu and FLAGS.tpu_name:
        tpu_cluster_resolver = tf.contrib.cluster_resolver.TPUClusterResolver(
            FLAGS.tpu_name, zone=FLAGS.tpu_zone, project=FLAGS.gcp_project)

    is_per_host = tf.contrib.tpu.InputPipelineConfig.PER_HOST_V2

    run_config = tf.contrib.tpu.RunConfig(
        cluster=tpu_cluster_resolver,
        master=FLAGS.master,
        model_dir=FLAGS.output_dir,
        save_checkpoints_steps=FLAGS.save_checkpoints_steps,
        tpu_config=tf.contrib.tpu.TPUConfig(
            iterations_per_loop=FLAGS.iterations_per_loop,
            num_shards=FLAGS.num_tpu_cores,
            per_host_input_for_training=is_per_host))

    num_train_steps = None
    num_warmup_steps = None

    # 返回的model_dn 是一个函数，其定义了模型，训练，评测方法,
    # 并且使用钩子参数，加载了BERT模型的参数进行了自己模型的参数初始化过程;
    # tf 新的架构方法，通过定义model_fn 函数，定义模型,
    # 然后通过EstimatorAPI进行模型的其他工作，Es就可以控制模型的训练，预测，评估工作等。

    model_fn = model_fn_builder(
        bert_config=bert_config,
        num_labels=len(label_list) + 1,
        init_checkpoint=FLAGS.init_checkpoint,
        learning_rate=FLAGS.learning_rate,
        num_train_steps=num_train_steps,
        num_warmup_steps=num_warmup_steps,
        use_tpu=FLAGS.use_tpu,
        use_one_hot_embeddings=FLAGS.use_tpu)

    estimator = tf.contrib.tpu.TPUEstimator(
        use_tpu=FLAGS.use_tpu,
        model_fn=model_fn,
        config=run_config,
        train_batch_size=FLAGS.train_batch_size,
        eval_batch_size=FLAGS.eval_batch_size,
        predict_batch_size=FLAGS.predict_batch_size)

    if FLAGS.do_predict:
        if FLAGS.raw_input:
            raw_input = input('Please input your text to extract entities:')
            lines_predict = list(raw_input.replace(' ', "O").strip())
            label_predict = ['O'] * len(lines_predict)
            lines = []
            l = ' '.join([label for label in label_predict if len(label) > 0])
            w = ' '.join([word for word in lines_predict if len(word) > 0])
            lines.append([l,w])
            predict_examples = processor._create_example(lines, 'test')
        else:
            Txt2Seq(FLAGS.data_dir, 'sample.txt')
            predict_examples = processor.get_predict_examples(FLAGS.data_dir)

        with open(FLAGS.output_dir + '/label2id.pkl', 'rb') as rf:
            label2id = pickle.load(rf)
            id2label = {value: key for key, value in label2id.items()}

        predict_file = os.path.join(FLAGS.data_dir, "predict.tf_record")
        batch_tokens, batch_labels = file_based_convert_examples_to_features(
            predict_examples, label_list, FLAGS.max_seq_length, tokenizer, predict_file, mode="test")

        logging.info("***** Running prediction*****")
        logging.info("  Num examples = %d", len(predict_examples))
        logging.info("  Batch size = %d", FLAGS.predict_batch_size)

        predict_drop_remainder = True if FLAGS.use_tpu else False
        if FLAGS.use_tpu:
            raise ValueError("Prediction in TPU not supported")

        predict_input_fn = file_based_input_fn_builder(
            input_file=predict_file,
            seq_length=FLAGS.max_seq_length,
            is_training=False,
            drop_remainder=predict_drop_remainder)

        result = estimator.predict(input_fn=predict_input_fn)

        if FLAGS.bilstm:
            predictions  = []
            for _, pred in enumerate(result):
                predictions.extend(pred)
        else:
            predictions = result

        output_predict_file = os.path.join(FLAGS.data_dir, "label_predict.txt")
        token_seq = []
        label_seq = []
        with open(output_predict_file,'w') as wf:
            for i, prediction in enumerate(predictions):
                token = batch_tokens[i]
                if prediction == 0:
                    continue
                predict = id2label[prediction]
                if token in ['[CLS]', '[SEP]']:
                    continue
                token_seq.append(token)
                label_seq.append(predict)
                line = "{}\t{}\n".format(token, predict)
                wf.write(line)

        seq2entity = Seq2Entity(token_seq, label_seq)
        per = seq2entity.get_per_entity()
        loc = seq2entity.get_loc_entity()
        org = seq2entity.get_org_entity()
        output_per = os.path.join(FLAGS.data_dir, 'per.txt')
        output_loc = os.path.join(FLAGS.data_dir, 'loc.txt')
        output_org = os.path.join(FLAGS.data_dir, 'org.txt')
        np.savetxt(output_per, np.array(per), fmt="%s")
        np.savetxt(output_loc, np.array(loc), fmt="%s")
        np.savetxt(output_org, np.array(org), fmt="%s")


if __name__ == "__main__":
    flags.mark_flag_as_required("data_dir")
    flags.mark_flag_as_required("vocab_file")
    flags.mark_flag_as_required("bert_config_file")
    flags.mark_flag_as_required("output_dir")
    LOG_SETTINGS = {
        'format': '%(asctime)s %(filename)s[line:%(lineno)d] %(levelname)s %(message)s',
        'datefmt': '%Y-%m-%d %H:%M:%S',
    }
    FILE_NAME = os.path.join(FLAGS.output_dir, 'predict.log')
    logging.basicConfig(
        handlers=[logging.FileHandler(FILE_NAME, encoding="utf-8", mode='a')],
        level=logging.INFO,
        **LOG_SETTINGS
    )
    tf.app.run()
