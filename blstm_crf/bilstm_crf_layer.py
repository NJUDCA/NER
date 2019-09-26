import tensorflow as tf
from tensorflow.contrib import rnn
from tensorflow.contrib import crf
from tensorflow.contrib.layers.python.layers import initializers


class BiLSTM_CRF:
    def __init__(self, embedded_chars, hidden_unit, cell_type, num_layers, dropout_rate,
                 num_labels, max_seq_length, labels, lengths, is_training, crf_only=False):
        """
        BLSTM-CRF 网络
        :param embedded_chars: Fine-tuning embedding input
        :param hidden_unit: LSTM的隐含单元个数
        :param cell_type: RNN类型（LSTM OR GRU DICNN will be add in feature）
        :param num_layers: RNN的层数
        :param droupout_rate: droupout rate
        :param num_labels: 标签数量
        :param max_seq_length: 序列最大长度
        :param labels: 真实标签
        :param lengths: [batch_size] 每个batch下序列的真实长度
        :param is_training: 是否是训练过程
        """
        self.hidden_unit = hidden_unit
        self.droupout_rate = dropout_rate
        self.cell_type = cell_type
        self.num_layers = num_layers
        self.embedded_chars = embedded_chars
        self.max_seq_length = max_seq_length
        self.num_labels = num_labels
        self.labels = labels
        self.lengths = lengths
        self.embedding_size = embedded_chars.shape[-1].value
        self.is_training = is_training
        self.crf_only = crf_only

    def _construct_cell(self):
        cell = None
        if self.cell_type == 'lstm':
            cell = rnn.BasicLSTMCell(self.hidden_unit)
        elif self.cell_type == 'gru':
            cell = rnn.GRUCell(self.hidden_unit)

        if self.droupout_rate is not None:
            cell = rnn.DropoutWrapper(cell, output_keep_prob=self.droupout_rate)
        return cell

    def bilstm_layer(self):
        with tf.variable_scope('bilstm_layer'):
            cell_fw = self._construct_cell()
            cell_bw = self._construct_cell()

            if self.num_layers > 1:
                cell_fw = rnn.MultiRNNCell([cell_fw] * self.num_layers, state_is_tuple=True)
                cell_bw = rnn.MultiRNNCell([cell_fw] * self.num_layers, state_is_tuple=True)

            output, _ = tf.nn.bidirectional_dynamic_rnn(cell_fw, cell_bw,
                                                        self.embedded_chars, dtype=tf.float32)
            output = tf.concat(output, axis=2)
        return output

    def project_bilstm_layer(self, lstm_output):
        """
        hidden layer between lstm layer and logits
        :param lstm_outputs: [batch_size, num_steps, emb_size]
        :return: [batch_size, num_steps, num_tags]
        """
        # tf.contrib.layers.xavier_initializer
        # 初始化权重矩阵
        with tf.variable_scope('project_bilstm_layer'):
            W = tf.get_variable('W', shape=[self.hidden_unit * 2, self.hidden_unit],
                                dtype=tf.float32, initializer=initializers.xavier_initializer())
            b = tf.get_variable("b", shape=[self.hidden_unit], dtype=tf.float32,
                                    initializer=tf.zeros_initializer())
            output = tf.reshape(lstm_output, shape=[-1, self.hidden_unit * 2])
            hidden = tf.tanh(tf.nn.xw_plus_b(output, W, b))

        # project to score of tags
        with tf.variable_scope("logits"):
            W = tf.get_variable("W", shape=[self.hidden_unit, self.num_labels],
                                dtype=tf.float32, initializer=initializers.xavier_initializer())

            b = tf.get_variable("b", shape=[self.num_labels], dtype=tf.float32,
                                initializer=tf.zeros_initializer())

            # tf.nn.xw_plus_b(x, weights, biases.)
            # 相当于matmul(x, weights) + biases.
            pred = tf.nn.xw_plus_b(hidden, W, b)
        return tf.reshape(pred, [-1, self.max_seq_length, self.num_labels])

    def project_crf_layer(self, embedding_chars):
        """
        hidden layer between input layer and logits
        :param lstm_outputs: [batch_size, num_steps, emb_size]
        :return: [batch_size, num_steps, num_tags]
        """
        with tf.variable_scope("project_crf_layer"):
            with tf.variable_scope("logits"):
                W = tf.get_variable("W", shape=[self.embedding_size, self.num_labels],
                                    dtype=tf.float32, initializer=initializers.xavier_initializer())

                b = tf.get_variable("b", shape=[self.num_labels], dtype=tf.float32,
                                    initializer=tf.zeros_initializer())
                output = tf.reshape(self.embedded_chars,
                                    shape=[-1, self.embedding_size])  # [batch_size, embedding_size]
                pred = tf.tanh(tf.nn.xw_plus_b(output, W, b))
            return tf.reshape(pred, [-1, self.max_seq_length, self.num_labels])

    def crf_layer(self, logits):
        """
        calculate crf loss
        :param project_logits: [1, num_steps, num_tags]
        :return: scalar loss
        """
        with tf.variable_scope("crf_loss"):
            trans = tf.get_variable(
                "transitions",
                shape=[self.num_labels, self.num_labels],
                initializer=initializers.xavier_initializer())
            if self.labels is None:
                return None, trans
            else:
                log_likelihood, trans = tf.contrib.crf.crf_log_likelihood(
                    inputs=logits,
                    tag_indices=self.labels,
                    transition_params=trans,
                    sequence_lengths=self.lengths)
                return tf.reduce_mean(-log_likelihood), trans

    def add_bilstm_crf_layer(self):
        if self.is_training:
            self.embedded_chars = tf.nn.dropout(self.embedded_chars, self.droupout_rate)

        if self.crf_only:
            # project layer
            logits = self.project_crf_layer(self.embedded_chars)
        else:
            # bilstm_layer
            lstm_output = self.bilstm_layer()
            # project layer
            logits = self.project_bilstm_layer(lstm_output)
        # crf_layer
        loss, trans = self.crf_layer(logits)

        # CRF decode, pred_ids 是一条最大概率的标注路径
        pred_ids, _ = crf.crf_decode(potentials=logits, transition_params=trans, sequence_length=self.lengths)
        return (loss, logits, pred_ids)


