"""Created by PeterLee, on Dec. 17."""
import tensorflow as tf
import numpy as np
import os
import argparse
from blstm_crf.model import BiLSTM_CRF
from processor.data import read_corpus, random_embedding, build_embedding_source
from application.txt2seq import Txt2Seq
from application.seq2entity import Seq2Entity
import logging
import pickle

## Session configuration
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

## tags, BIO
tag2label = {
             "B-LOC": 0, "I-LOC": 1,
             "B-PER": 2, "I-PER": 3,
             "B-ORG": 4, "I-ORG": 5,
             "O": 6
             }


## hyperparameters setting
parser = argparse.ArgumentParser(description='BiLSTM-CRF for Chinese NER task')
parser.add_argument('--data_dir', type=str, default=None, help='The input dir.')
parser.add_argument('--output_dir', type=str, default=None, help='The output dir.')
parser.add_argument('--embedding_source', type=str, default=None, help='The embedding source file.')
parser.add_argument('--vocab_file', type=str, default='char2id.pkl', help='The vocab file.')
parser.add_argument('--embedding_file', type=str, default='char2vec.txt', help='pretrained char embedding file. if None, init it randomly')
parser.add_argument('--embedding_dim', type=int, default=100, help='random init char embedding_dim')
parser.add_argument('--random_embedding', type=bool, default=False, help='use random embedding or not')
parser.add_argument('--update_embedding', type=bool, default=True, help='update embedding during training')
parser.add_argument('--batch_size', type=int, default=64, help='sample of each minibatch')
parser.add_argument('--num_train_epochs', type=int, default=25, help='epoch of training')
parser.add_argument('--lstm_size', type=int, default=300, help='dim of hidden state')
parser.add_argument('--optimizer', type=str, default='Adam', help='Adam/Adadelta/Adagrad/RMSProp/Momentum/SGD')
parser.add_argument('--CRF', type=bool, default=True, help='use CRF at the top layer. if False, use Softmax')
parser.add_argument('--lr', type=float, default=0.001, help='learning rate')
parser.add_argument('--clip', type=float, default=5.0, help='gradient clipping')
parser.add_argument('--dropout', type=float, default=0.5, help='dropout keep_prob')
parser.add_argument('--shuffle', type=bool, default=True, help='shuffle training data before each epoch')
parser.add_argument('--mode', type=str, default='train', help='train/test/predict')
parser.add_argument('--ckpt_model', type=str, default='model.ckpt-123', help='model for demo')
parser.add_argument('--raw_input', type=bool, default=False, help='input text from console. If false, from file')
args = parser.parse_args()


def main():
    # prepare vocab and embedding
    if not os.path.exists(args.vocab_file) and not os.path.exists(args.embedding_file):
        build_embedding_source(args.embedding_source, args.vocab_file, args.embedding_file)
    with open(args.vocab_file, 'rb') as fr:
        word2id = pickle.load(fr)
    logging.info('size of vocabulary: {}'.format(len(word2id)))

    # get char embeddings
    if args.random_embedding:
        embeddings = random_embedding(word2id, args.embedding_dim)
    else:
        embeddings = np.loadtxt(args.embedding_file)
    logging.info('shape of char embeddings: {}'.format(embeddings.shape))
    
    # check output paths
    paths = dict()
    summary_path = os.path.join(args.output_dir, "summaries")
    if not os.path.exists(summary_path):
        os.makedirs(summary_path)
    paths['summary_path'] = summary_path
    model_path = os.path.join(args.output_dir, "checkpoints/")
    if not os.path.exists(model_path):
        os.makedirs(model_path)
    ckpt_prefix = os.path.join(model_path, "model.ckpt")
    paths['model_path'] = ckpt_prefix
    result_path = os.path.join(args.output_dir, "results/")
    if not os.path.exists(result_path):
        os.makedirs(result_path)
    paths['result_path'] = result_path
    
    if args.mode == 'train':
        ## read corpus and get training data
        train_data = read_corpus(os.path.join(args.data_dir, 'train.txt'))
        dev_data = read_corpus(os.path.join(args.data_dir, 'dev.txt'))
        logging.info('{} train examples, {} dev examples'.format(len(train_data), len(dev_data)))
        
        model = BiLSTM_CRF(args, embeddings, tag2label, word2id, paths)
        model.build_graph()
        model.train(train=train_data, dev=dev_data)
    elif args.mode == 'test':
        test_data = read_corpus(os.path.join(args.data_dir, 'dev.txt'))
        logging.info('{} test examples'.format(len(test_data)))

        ckpt_file = tf.train.latest_checkpoint(model_path)
        logging.info('last checkpoint file: {}'.format(ckpt_file))
        paths['model_path'] = ckpt_file
        model = BiLSTM_CRF(args, embeddings, tag2label, word2id, paths)
        model.build_graph()
        model.test(test_data)
    elif args.mode == 'predict':
        ckpt_file = args.ckpt_model if os.path.exists(args.ckpt_model) else tf.train.latest_checkpoint(model_path)
        logging.info('trained checkpoint file: {}'.format(ckpt_file))
        paths['model_path'] = ckpt_file
        model = BiLSTM_CRF(args, embeddings, tag2label, word2id, paths)
        model.build_graph()
        saver = tf.train.Saver()
        with tf.Session() as sess:
            saver.restore(sess, ckpt_file)
            predict_data = []
            token_list = []
            if args.raw_input:
                raw_input = input('Please input your text to extract entities:')
                token_list = list(raw_input.replace(' ', "O").strip())
                tag_list = ['O'] * len(token_list)
                predict_data.append((token_list, tag_list))
            else:
                Txt2Seq(args.data_dir, 'demo.txt')
                predict_data = read_corpus(os.path.join(args.data_dir, 'predict.txt'))
                for (token_, _) in predict_data:
                    token_list.extend(token_)
            logging.info('{} predict examples'.format(len(predict_data)))
            tag_predict = model.demo_one(sess, predict_data)
            logging.info('tokens: {}'.format(' '.join([str(token) for token in token_list])))
            logging.info('tags: {}'.format(' '.join([str(tag) for tag in tag_predict])))

            seq2entity = Seq2Entity(token_list, tag_predict)
            per = seq2entity.get_per_entity()
            loc = seq2entity.get_loc_entity()
            org = seq2entity.get_org_entity()
            return {
                'per': per,
                'loc': loc,
                'org': org
            }



if __name__ == "__main__":
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)
    LOG_SETTINGS = {
        'format': '%(asctime)s %(filename)s[line:%(lineno)d] %(levelname)s %(message)s',
        'datefmt': '%Y-%m-%d %H:%M:%S',
    }
    FILE_NAME = os.path.join(args.output_dir, '{}.log'.format(args.mode))
    logging.basicConfig(
        handlers=[logging.FileHandler(FILE_NAME, encoding="utf-8", mode='a')],
        level=logging.INFO,
        **LOG_SETTINGS
    )
    logging.info(str(args))
    main()
    

    
    
    