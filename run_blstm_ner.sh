#!/usr/bin/env bash

# train and eval in source domain
# python blstm_ner.py\
#     --data_dir=./data/ChinaDaily   \
#     --output_dir=./output/ChinaDaily/blstm_crf/   \
#     --vocab_file=./data/wiki/char2id.pkl   \
#     --embedding_file=./data/wiki/char2vec.txt   \
#     --embedding_source=./data/wiki/wiki_100.utf8.txt   \
#     --lr=0.001   \
#     --num_train_epochs=40   \
#     --batch_size=64   \
#     --CRF=True   \
#     --mode=train

# predict in target domain
# python blstm_ner.py\
#     --data_dir=./data/ywsz/   \
#     --output_dir=./output/ChinaDaily/blstm_crf/   \
#     --vocab_file=./data/wiki/char2id.pkl   \
#     --embedding_file=./data/wiki/char2vec.txt   \
#     --CRF=True   \
#     --mode=predict   \

# test in target domain
# using char2vec blstm_crf
# python blstm_ner.py\
#     --data_dir=./data/ywevents/   \
#     --output_dir=./output/ChinaDaily/blstm_crf/   \
#     --vocab_file=./data/wiki/char2id.pkl   \
#     --embedding_file=./data/wiki/char2vec.txt   \
#     --CRF=True   \
#     --mode=test   \

# test in target domain
# using random blstm_crf
python blstm_ner.py\
    --data_dir=./data/ywevents/   \
    --output_dir=./output/ChinaDaily/blstm_crf_random/   \
    --vocab_file=./data/wiki/char2id.pkl   \
    --embedding_file=./data/wiki/char2vec.txt   \
    --CRF=True   \
    --random_embedding=True   \
    --mode=test   \

