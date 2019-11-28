import numpy as np
import os
import sys
sys.path.append("..")
import tensorflow as tf
from tensorflow.contrib import rnn
from tensorflow.contrib import crf
from processor.data import pad_sequences, batch_yield
from sklearn.metrics import classification_report
from metrics import conlleval
import logging


class BiLSTM_CRF(object):
    def __init__(self, args, embeddings, tag2label, vocab, paths):
        self.batch_size = args.batch_size
        self.num_train_epochs = args.num_train_epochs
        self.lstm_size = args.lstm_size
        self.embeddings = embeddings
        self.CRF = args.CRF
        self.update_embedding = args.update_embedding
        self.dropout_keep_prob = args.dropout
        self.optimizer = args.optimizer
        self.lr = args.lr
        self.clip_grad = args.clip
        self.tag2label = tag2label
        self.num_tags = len(tag2label)
        self.vocab = vocab  # word2id(.pkl)
        self.shuffle = args.shuffle
        self.model_path = paths['model_path']
        self.summary_path = paths['summary_path']  # Path to save the summary
        self.result_path = paths['result_path'] 

    def build_graph(self):
        # Creating the input layer
        self.add_placeholders()
        # Creating the lookup layer
        self.lookup_layer_op()
        # Creating bi-LSTM+CRF layer
        self.biLSTM_layer_op()
        # If doesn't set crf, set softmax layer
        self.softmax_pred_op()
        # Setting loss function
        self.loss_op()
        # Training
        self.trainstep_op()
        # Initializing parms
        self.init()

    def add_placeholders(self):
        # parameters initialization
        self.word_ids = tf.placeholder(tf.int32, shape=[None, None], name="word_ids")
        self.labels = tf.placeholder(tf.int32, shape=[None, None], name="labels")
        self.sequence_lengths = tf.placeholder(tf.int32, shape=[None], name="sequence_lengths")
        self.dropout_pl = tf.placeholder(dtype=tf.float32, shape=[], name="dropout")
        self.lr_pl = tf.placeholder(dtype=tf.float32, shape=[], name="lr")

    def lookup_layer_op(self):
        with tf.variable_scope("words"):
            _word_embeddings = tf.Variable(self.embeddings, dtype=tf.float32, trainable=self.update_embedding,
                                           name="_word_embeddings")
            word_embeddings = tf.nn.embedding_lookup(params=_word_embeddings, ids=self.word_ids, name="word_embeddings")
        self.word_embeddings = tf.nn.dropout(word_embeddings, self.dropout_pl)

    def biLSTM_layer_op(self):
        with tf.variable_scope("bi-lstm"):
            cell_fw = rnn.LSTMCell(self.lstm_size)
            cell_bw = rnn.LSTMCell(self.lstm_size)
            # bi-lstm
            (output_fw_seq, output_bw_seq), _ = tf.nn.bidirectional_dynamic_rnn(cell_fw=cell_fw, cell_bw=cell_bw,
                                                                                inputs=self.word_embeddings,
                                                                                sequence_length=self.sequence_lengths,
                                                                                dtype=tf.float32)
            output = tf.concat([output_fw_seq, output_bw_seq], axis=-1)
            output = tf.nn.dropout(output, self.dropout_pl)

        with tf.variable_scope("proj"):
            W = tf.get_variable(name="W", shape=[2 * self.lstm_size, self.num_tags],
                                initializer=tf.contrib.layers.xavier_initializer(), dtype=tf.float32)
            b = tf.get_variable(name="b", shape=[self.num_tags], initializer=tf.zeros_initializer(), dtype=tf.float32)
            s = tf.shape(output)
            output = tf.reshape(output, [-1, 2 * self.lstm_size])
            pred = tf.matmul(output, W) + b
            # crf
            self.logits = tf.reshape(pred, [-1, s[1], self.num_tags])

    def loss_op(self):
        if self.CRF:
            # log_likelihood loss function
            log_likelihood, self.transition_params = crf.crf_log_likelihood(inputs=self.logits, tag_indices=self.labels,
                                                                            sequence_lengths=self.sequence_lengths)
            self.loss = -tf.reduce_mean(log_likelihood)
        else:
            # not crf, log cross entropy loss function
            losses = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=self.logits, labels=self.labels)
            mask = tf.sequence_mask(self.sequence_lengths)
            losses = tf.boolean_mask(losses, mask)
            self.loss = tf.reduce_mean(losses)
        # Recording the loss value in summary
        tf.summary.scalar("loss", self.loss)

    def softmax_pred_op(self):
        if not self.CRF:
            self.labels_softmax_ = tf.argmax(self.logits, axis=-1)
            self.labels_softmax_ = tf.cast(self.labels_softmax_, tf.int32)

    def trainstep_op(self):
        with tf.variable_scope("train_step"):
            self.global_step = tf.Variable(0, name="global_step", trainable=False)
            # Setting optimizing method
            if self.optimizer == 'Adam':
                optim = tf.train.AdamOptimizer(learning_rate=self.lr_pl)
            elif self.optimizer == 'Adadelta':
                optim = tf.train.AdadeltaOptimizer(learning_rate=self.lr_pl)
            elif self.optimizer == 'Adagrad':
                optim = tf.train.AdagradOptimizer(learning_rate=self.lr_pl)
            elif self.optimizer == 'RMSProp':
                optim = tf.train.RMSPropOptimizer(learning_rate=self.lr_pl)
            elif self.optimizer == 'Momentum':
                optim = tf.train.MomentumOptimizer(learning_rate=self.lr_pl, momentum=0.9)
            elif self.optimizer == 'SGD':
                optim = tf.train.GradientDescentOptimizer(learning_rate=self.lr_pl)
            else:
                optim = tf.train.GradientDescentOptimizer(learning_rate=self.lr_pl)
            # Compute gradients of loss for the variables in var_list,
            # Return:A list of (gradient, variable) pairs. Variable is always present, but gradient can be None
            grads_and_vars = optim.compute_gradients(self.loss)
            # Clips tensor values to a specified min and max, min=-self.clip_grad, max=self.clip_grad
            grads_and_vars_clip = [[tf.clip_by_value(g, -self.clip_grad, self.clip_grad), v] for g, v in grads_and_vars]
            # Apply gradients to variables
            self.train_op = optim.apply_gradients(grads_and_vars_clip, global_step=self.global_step)

    def init(self):
        self.init_op = tf.global_variables_initializer()

    def add_summary(self, sess):
        # Merges all summaries collected in the default graph
        self.merged = tf.summary.merge_all()
        # Writes Summary protocol buffers to event files
        self.file_writer = tf.summary.FileWriter(self.summary_path, sess.graph)

    def train(self, train, dev):
        """
        :param train: training data
        :param dev: testing data
        :return:
        """

        # Creating a saver
        saver = tf.train.Saver(tf.global_variables())
        # Defining a sessinon
        with tf.Session() as sess:
            # Parameters initialization
            sess.run(self.init_op)
            # Wrinting the information to summary
            self.add_summary(sess)
            for epoch in range(self.num_train_epochs):
                logging.info('==========training==========')
                self.run_one_epoch(sess, train, dev, self.tag2label, epoch, saver)

    def test(self, test):
        saver = tf.train.Saver()
        with tf.Session() as sess:
            logging.info('=========== testing ===========')
            saver.restore(sess, self.model_path)
            label_list, _ = self.dev_one_epoch(sess, test)
            self.evaluate(label_list, test)
            # label2tag = {}
            # for tag, label in self.tag2label.items():
                # label2tag[label] = tag
            # tag_list = []
            # for label_ in label_list:
                # tag_list.append([label2tag[label] for label in label_])
            # test_label_file = os.path.join(self.result_path, 'test_label.txt')
            # test_result_file = os.path.join(self.result_path, 'test_result.txt')
            # with open(test_label_file, 'w+', encoding='utf-8') as fw:
                # for tags_, (sent, tags) in zip(tag_list, test):
                    # if len(tags_) != len(sent):
                        # print(sent)
                        # print(len(tags_))
                    # else:
                        # for token, true_tag, predict_tag in zip(sent, tags, tags_):
                            # fw.write("{}\t{}\t{}\n".format(token, true_tag, predict_tag))
            # test_report = conlleval.return_report(test_label_file)
            # logging.info(''.join(test_report))
            # with open(test_result_file, 'a+', encoding='UTF-8') as wf:
                # wf.write(''.join(test_report))


    def demo_one(self, sess, sent):
        label_list = []
        for seqs, _ in batch_yield(sent, self.batch_size, self.vocab, self.tag2label, shuffle=False):
            label_list_, _ = self.predict_one_batch(sess, seqs)
            label_list.extend(label_list_)
        label2tag = {}
        for tag, label in self.tag2label.items():
            label2tag[label] = tag
        tag_list = []
        for label_ in label_list:
            tag_list.append([label2tag[label] for label in label_])
        return tag_list

    def run_one_epoch(self, sess, train, dev, tag2label, epoch, saver):
        num_batches = (len(train) + self.batch_size - 1) // self.batch_size
        # Calling function batch_yield
        batches = batch_yield(train, self.batch_size, self.vocab, self.tag2label, shuffle=self.shuffle)
        # Creating a index list, step is index and (seqs, labels) is data
        for step, (seqs, labels) in enumerate(batches):

            sys.stdout.write(' processing: {} batch / {} batches.'.format(step + 1, num_batches) + '\r')
            step_num = epoch * num_batches + step + 1
            feed_dict, _ = self.get_feed_dict(seqs, labels, self.lr, self.dropout_keep_prob)
            _, loss_train, summary, step_num_ = sess.run([self.train_op, self.loss, self.merged, self.global_step],
                                                         feed_dict=feed_dict)
            if step + 1 == 1 or (step + 1) % 300 == 0 or step + 1 == num_batches:
                # Writing the log information
                logging.info('epoch {}, step {}, loss: {:.4}, global_step: {}'.format(epoch + 1, step + 1, loss_train, step_num))
            # Writing the information(loss) to summary
            self.file_writer.add_summary(summary, step_num)

            if step + 1 == num_batches:
                saver.save(sess, self.model_path, global_step=step_num)

        logging.info('==========validation ==========')
        label_list_dev, seq_len_list_dev = self.dev_one_epoch(sess, dev)
        self.evaluate(label_list_dev, dev, epoch)
        logging.info('epoch {} finished'.format(epoch + 1))

    def get_feed_dict(self, seqs, labels=None, lr=None, dropout=None):
        # Calling the function of pad_sequences
        word_ids, seq_len_list = pad_sequences(seqs, pad_mark=0)
        # Feeding the data
        feed_dict = {self.word_ids: word_ids, self.sequence_lengths: seq_len_list}
        if labels is not None:
            # Calling the function of pad_sequences
            labels_, _ = pad_sequences(labels, pad_mark=0)
            # Feeding the label
            feed_dict[self.labels] = labels_
        if lr is not None:
            # Feeding the learning rate
            feed_dict[self.lr_pl] = lr
        if dropout is not None:
            # Feeding the dropout
            feed_dict[self.dropout_pl] = dropout
        return feed_dict, seq_len_list

    def dev_one_epoch(self, sess, dev):
        """
        :param sess: Session
        :param dev: testing data
        :return:
        """
        label_list, seq_len_list = [], []
        for seqs, _ in batch_yield(dev, self.batch_size, self.vocab, self.tag2label, shuffle=False):
            label_list_, seq_len_list_ = self.predict_one_batch(sess, seqs)
            label_list.extend(label_list_)
            seq_len_list.extend(seq_len_list_)
        return label_list, seq_len_list

    def predict_one_batch(self, sess, seqs):
        feed_dict, seq_len_list = self.get_feed_dict(seqs, dropout=1.0)
        if self.CRF:
            logits, transition_params = sess.run([self.logits, self.transition_params],
                                                 feed_dict=feed_dict)
            label_list = []
            for logit, seq_len in zip(logits, seq_len_list):
                # Calling the viterbi algorithm to find the best results(probability highest)
                # Decode the highest scoring sequence of tags(this should only be used at test time)
                # Return(viterbi_seq): a [seq_len] list of integers containing the highest scoring tag indices
                viterbi_seq, _ = crf.viterbi_decode(logit[:seq_len], transition_params)
                label_list.append(viterbi_seq)
            return label_list, seq_len_list

        else:
            # Softmax
            logits = sess.run(self.labels_softmax_, feed_dict=feed_dict)
            label_list = []
            for logit, seq_len in zip(logits, seq_len_list):
                label_list.append(logit[:seq_len])
            return label_list, seq_len_list

    def evaluate(self, label_list, data, epoch=None):
        label_true = []
        label_pred = []
        report = None
        target_names = []
        for key, _ in sorted(self.tag2label.items(), key=lambda item: item[1]):
            target_names.append(key)
        for labels_, (sent, tags) in zip(label_list, data):
            labels = [self.tag2label[tag] for tag in tags ]
            if len(labels_) != len(sent):
                print(sent)
                print(len(labels_))
            else:
                label_pred.extend(labels_)
                label_true.extend(labels)
        report = classification_report(label_true, label_pred, target_names=target_names, digits=4)
        
        epoch_num = str(epoch + 1) if epoch != None else 'test'
        logging.info('classification_report for {}\n{}'.format(epoch_num, report))
        eval_file = os.path.join(self.result_path, 'eval_result.txt')
        with open(eval_file, 'a+') as fw:
            fw.write('epoch_num: {}\n'.format(epoch_num))
            fw.write('classification_report: \n{}\n'.format(report))