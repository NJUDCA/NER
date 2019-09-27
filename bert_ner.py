#! usr/bin/env python3
# -*- coding:utf-8 -*-
"""
Copyright 2018 The Google AI Language Team Authors.
BASED ON Google_BERT.
@Author:zhoukaiyin
Adjust code for chinese ner
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import collections
import os
from bert import modeling
from bert import optimization
from bert import tokenization
from blstm_crf.bilstm_crf_layer import BiLSTM_CRF
from metrics import metrics_cm
import tensorflow as tf
import pickle
import logging


# ML的模型中有大量需要tuning的超参数
# flags可以帮助我们通过命令行来动态的更改代码中的参数
# 类似于 argparse
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

flags.DEFINE_bool("do_train", True,
                  "Whether to run training.")

flags.DEFINE_bool("do_eval", True,
                  "Whether to run eval on the dev set.")

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

flags.DEFINE_float("num_train_epochs", 3.0,
                   "Total number of training epochs to perform.")

flags.DEFINE_float("warmup_proportion", 0.1,
                   "Proportion of training to perform linear learning rate warmup for. "
                   "E.g., 0.1 = 10% of training.")

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

LOG_SETTINGS = {
    'format': '%(asctime)s %(filename)s[line:%(lineno)d] %(levelname)s %(message)s',
    'datefmt': '%Y-%m-%d %H:%M:%S',
    'filename': FLAGS.output_dir + 'bert_ner.log',
    'filemode': 'a',
}
logging.basicConfig(level=logging.INFO, **LOG_SETTINGS)


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

    def get_test_examples(self, data_dir):
        return self._create_example(
            self._read_data(os.path.join(data_dir, "test.txt")), "test")

    def get_labels(self):
        '''
        'X' is used to represent "##eer","##soo" and so on!
        "[PAD]" for padding
        '''
        # return ["[PAD]", "X", "B", "I", "E", "S", "O", "[CLS]", "[SEP]"]
        # based on ChinaDaily corpus
        return ["[PAD]", "X", "B-LOC", "I-LOC", "B-PER", "I-PER", "B-ORG", "I-ORG", "O", "[CLS]", "[SEP]"]


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
    for i, label in enumerate(label_list):
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
    if len(tokens) >= max_seq_length - 1:
        tokens = tokens[0: max_seq_length - 1]
        labels = labels[0: max_seq_length - 1]
    ntokens = []
    segment_ids = []
    label_ids = []
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
    # ntokens.append("[SEP]")
    # segment_ids.append(0)
    # # append("O") or append("[SEP]") not sure!
    # label_ids.append(label_map["[SEP]"])

    input_ids = tokenizer.convert_tokens_to_ids(ntokens)
    input_mask = [1] * len(input_ids)
    #use zero to padding sequence
    while len(input_ids) < max_seq_length:
        input_ids.append(0)
        input_mask.append(0)
        segment_ids.append(0)
        label_ids.append(0)
        ntokens.append("[PAD]")
    assert len(input_ids) == max_seq_length
    assert len(input_mask) == max_seq_length
    assert len(segment_ids) == max_seq_length
    assert len(label_ids) == max_seq_length

    if ex_index < 2:
        logging.info("*** Example ***")
        logging.info("guid: %s" % (example.guid))
        logging.info("tokens: %s" % " ".join(
            [tokenization.printable_text(x) for x in tokens]))
        logging.info("input_ids: %s" % " ".join([str(x) for x in input_ids]))
        logging.info("input_mask: %s" % " ".join([str(x) for x in input_mask]))
        logging.info("segment_ids: %s" % " ".join([str(x) for x in segment_ids]))
        logging.info("label_ids: %s" % " ".join([str(x) for x in label_ids]))

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
    model = modeling.BertModel(
        config=bert_config,
        is_training=is_training,
        input_ids=input_ids,
        input_mask=input_mask,
        token_type_ids=segment_ids,
        use_one_hot_embeddings=use_one_hot_embeddings
    )
    
    output_layer = model.get_sequence_output()

    if FLAGS.bilstm:
        # 获取相应的output作为emedding
        # input data [batch_size, seq_length, embedding_size]
        max_seq_length = output_layer.shape[1].value
        used = tf.sign(tf.abs(input_ids))
        lengths = tf.reduce_sum(used, reduction_indices=1)  # [batch_size] 大小的向量，包含了当前batch中的序列长度

        bilstm_crf = BiLSTM_CRF(embedded_chars=output_layer, hidden_unit=FLAGS.lstm_size, cell_type=FLAGS.cell, num_layers=FLAGS.num_layers, 
                                dropout_rate=FLAGS.dropout_rate, num_labels=num_labels, max_seq_length=max_seq_length, labels=labels,
                                lengths=lengths, is_training=is_training, crf_only=FLAGS.crf_only)
        loss, logits, predict = bilstm_crf.add_bilstm_crf_layer()
    else:
        if is_training:
            output_layer = tf.nn.dropout(output_layer, FLAGS.dropout_rate)
        logits = hidden2tag(output_layer, num_labels)
        logits = tf.reshape(logits, [-1, FLAGS.max_seq_length, num_labels])
        loss, predict = softmax_layer(logits, labels, num_labels, input_mask)
        return (loss, logits, predict)


def model_fn_builder(bert_config, num_labels, init_checkpoint, learning_rate,
                     num_train_steps, num_warmup_steps, use_tpu, use_one_hot_embeddings):
    def model_fn(features, labels, mode, params):
        logging.info("*** Features ***")
        for name in sorted(features.keys()):
            tf.logging.info("  name = {}, shape = {}".format(name, features[name].shape))
        input_ids = features["input_ids"]
        input_mask = features["input_mask"]
        segment_ids = features["segment_ids"]
        label_ids = features["label_ids"]
        is_training = (mode == tf.estimator.ModeKeys.TRAIN)

        (total_loss, logits, predicts) = create_model(
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
        if mode == tf.estimator.ModeKeys.TRAIN:
            train_op = optimization.create_optimizer(
                total_loss, learning_rate, num_train_steps, num_warmup_steps, use_tpu)
            output_spec = tf.contrib.tpu.TPUEstimatorSpec(
                mode=mode,
                loss=total_loss,
                train_op=train_op,
                scaffold_fn=scaffold_fn)
        elif mode == tf.estimator.ModeKeys.EVAL:
            # 针对NER ,进行了修改
            def metric_fn(label_ids, logits, num_labels, input_mask):
                predictions = tf.argmax(logits, axis=-1, output_type=tf.int32)
                cm = metrics_cm.streaming_confusion_matrix(label_ids, predictions, num_labels - 1, weights=input_mask)
                return {
                    "confusion_matrix": cm
                }
            eval_metrics = (metric_fn, [label_ids, logits, num_labels, input_mask])

            output_spec = tf.contrib.tpu.TPUEstimatorSpec(
                mode=mode, loss=total_loss, eval_metrics=eval_metrics, scaffold_fn=scaffold_fn
            )
        else:
            output_spec = tf.contrib.tpu.TPUEstimatorSpec(
                mode=mode, predictions=predicts, scaffold_fn=scaffold_fn)
        return output_spec

    return model_fn


def Writer(output_predict_file, result, batch_tokens, batch_labels, id2label):
    with open(output_predict_file,'w') as wf:
        
        if FLAGS.bilstm:
            predictions  = []
            for _, pred in enumerate(result):
                predictions.extend(pred)
        else:
            predictions = result

        for i, prediction in enumerate(predictions):
            token = batch_tokens[i]
            predict = id2label[prediction]
            # true_label = id2label[batch_labels[i]]
            if token!="[PAD]" and token!="[CLS]" and true_label !="X":

                if predict=="X" and not predict.startswith("##"):
                    predict="O"
                # line = "{}\t{}\t{}\n".format(token, true_label, predict)
                line = "{}\t{}\n".format(token, predict)
                wf.write(line)


def main(_):
    # tf.logging.set_verbosity(tf.logging.INFO)

    logging.info(str(FLAGS))

    processors = {
        "ner": NerProcessor
    }
    if not FLAGS.do_train and not FLAGS.do_eval:
        raise ValueError("At least one of `do_train` or `do_eval` must be True.")

    bert_config = modeling.BertConfig.from_json_file(FLAGS.bert_config_file)

    if FLAGS.max_seq_length > bert_config.max_position_embeddings:
        raise ValueError("Cannot use sequence length {} because the BERT model, was only trained up to sequence length {}".format(
            FLAGS.max_seq_length, bert_config.max_position_embeddings))

    # tf.io.gfile.makedirs(FLAGS.output_dir)
    tf.gfile.MakeDirs(FLAGS.output_dir)

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

    train_examples = None
    num_train_steps = None
    num_warmup_steps = None

    if FLAGS.do_train:
        train_examples = processor.get_train_examples(FLAGS.data_dir)
        num_train_steps = int(
            len(train_examples) / (FLAGS.train_batch_size * FLAGS.num_train_epochs))
        num_warmup_steps = int(num_train_steps * FLAGS.warmup_proportion)
    
    # 返回的model_dn 是一个函数，其定义了模型，训练，评测方法,
    # 并且使用钩子参数，加载了BERT模型的参数进行了自己模型的参数初始化过程;
    # tf 新的架构方法，通过定义model_fn 函数，定义模型,
    # 然后通过EstimatorAPI进行模型的其他工作，Es就可以控制模型的训练，预测，评估工作等。

    model_fn = model_fn_builder(
        bert_config=bert_config,
        num_labels=len(label_list),
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

    if FLAGS.do_train:
        train_file = os.path.join(FLAGS.output_dir, "train.tf_record")
        print('***train_file***', train_file)
        _, _ = file_based_convert_examples_to_features(
            train_examples, label_list, FLAGS.max_seq_length, tokenizer, train_file)
        logging.info("***** Running training *****")
        logging.info("  Num examples = %d", len(train_examples))
        logging.info("  Batch size = %d", FLAGS.train_batch_size)
        logging.info("  Num steps = %d", num_train_steps)
        train_input_fn = file_based_input_fn_builder(
            input_file=train_file,
            seq_length=FLAGS.max_seq_length,
            is_training=True,
            drop_remainder=True)
        estimator.train(input_fn=train_input_fn, max_steps=num_train_steps)

    if FLAGS.do_eval:
        eval_examples = processor.get_dev_examples(FLAGS.data_dir)
        eval_file = os.path.join(FLAGS.output_dir, "eval.tf_record")
        _, _ = file_based_convert_examples_to_features(
            eval_examples, label_list, FLAGS.max_seq_length, tokenizer, eval_file)

        logging.info("***** Running evaluation *****")
        logging.info("  Num examples = %d", len(eval_examples))
        logging.info("  Batch size = %d", FLAGS.eval_batch_size)

        eval_steps = None
        eval_drop_remainder = False
        if FLAGS.use_tpu:
            eval_steps = int(len(eval_examples) / FLAGS.eval_batch_size)
            eval_drop_remainder = True

        eval_input_fn = file_based_input_fn_builder(
            input_file=eval_file,
            seq_length=FLAGS.max_seq_length,
            is_training=False,
            drop_remainder=eval_drop_remainder)

        result = estimator.evaluate(input_fn=eval_input_fn, steps=eval_steps)
        output_eval_file = os.path.join(FLAGS.output_dir, "eval_results.txt")

        with open(output_eval_file, "w") as writer:
            logging.info("***** Eval results *****")
            confusion_matrix = result["confusion_matrix"]
            p, r, f = metrics_cm.calculate(confusion_matrix,len(label_list)-1)
            logging.info("P = %s",  str(p))
            logging.info("R = %s",  str(r))
            logging.info("F = %s",  str(f))
            writer.write("P = {}\nR = {}\nF = {}".format(p, r, f))

    if FLAGS.do_predict:
        with open(FLAGS.output_dir + '/label2id.pkl', 'rb') as rf:
            label2id = pickle.load(rf)
            id2label = {value: key for key, value in label2id.items()}
        
        token_path = os.path.join(FLAGS.output_dir, "token_test.txt")
        if os.path.exists(token_path):
            os.remove(token_path)

        predict_examples = processor.get_test_examples(FLAGS.data_dir)
        predict_file = os.path.join(FLAGS.output_dir, "predict.tf_record")
        batch_tokens, batch_labels = file_based_convert_examples_to_features(
            predict_examples, label_list, FLAGS.max_seq_length, tokenizer, predict_file, mode="test")

        logging.info("***** Running prediction*****")
        logging.info("  Num examples = %d", len(predict_examples))
        logging.info("  Batch size = %d", FLAGS.predict_batch_size)

        predict_drop_remainder = True
        if FLAGS.use_tpu:
            predict_drop_remainder = False
            # Warning: According to tpu_estimator.py Prediction on TPU is an
            # experimental feature and hence not supported here
            raise ValueError("Prediction in TPU not supported")

        predict_input_fn = file_based_input_fn_builder(
            input_file=predict_file,
            seq_length=FLAGS.max_seq_length,
            is_training=False,
            drop_remainder=predict_drop_remainder)

        result = estimator.predict(input_fn=predict_input_fn)
        output_predict_file = os.path.join(FLAGS.output_dir, "label_test.txt")
        # here if the tag is "X" means it belong to its before token, here for convenient evaluate use
        # conlleval.pl we  discarding it directly
        Writer(output_predict_file, result, batch_tokens, batch_labels, id2label)


if __name__ == '__main__':
    flags.mark_flag_as_required("data_dir")
    flags.mark_flag_as_required("task_name")
    flags.mark_flag_as_required("vocab_file")
    flags.mark_flag_as_required("bert_config_file")
    flags.mark_flag_as_required("output_dir")
    tf.app.run()
    # tf.compat.v1.app.run()
