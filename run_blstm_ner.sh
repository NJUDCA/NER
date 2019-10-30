#!/usr/bin/env bash

# train and eval
python blstm_ner.py\
    --data_dir=./data/ChinaDaily   \
    --output_dir=./output/ChinaDaily/blstm_crf/   \
    --vocab_file=./data/wiki/char2id.pkl   \
    --embedding_file=./data/wiki/char2vec.txt   \
    --embedding_source=./data/wiki/wiki_100.utf8.txt   \
    --lr=0.001   \
    --num_train_epochs=40   \
    --batch_size=64   \
    --CRF=True   \
    --mode=train

# predict
python blstm_ner.py\
    --data_dir=./data/ywsz/   \
    --output_dir=./output/ChinaDaily/blstm_crf/   \
    --vocab_file=./data/wiki/char2id.pkl   \
    --embedding_file=./data/wiki/char2vec.txt   \
    --CRF=True   \
    --mode=predict   \

