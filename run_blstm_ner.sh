#!/usr/bin/env bash

# train and eval in source domain
# python blstm_ner.py\
#     --data_dir=./data/MSRA   \
#     --output_dir=./output/MSRA/blstm_crf/   \
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
#     --output_dir=./output/MSRA/blstm_crf/   \
#     --vocab_file=./data/wiki/char2id.pkl   \
#     --embedding_file=./data/wiki/char2vec.txt   \
#     --CRF=True   \
#     --mode=predict   \


# test in target domain
# using blstm
# python blstm_ner.py\
    # --data_dir=./data/ywevents/   \
    # --output_dir=./output/ywevents/blstm/   \
    # --ckpt_model=./output/MSRA/blstm/checkpoints/model.ckpt-31680   \
    # --vocab_file=./data/wiki/char2id.pkl   \
    # --embedding_file=./data/wiki/char2vec.txt   \
    # --random_embedding=True   \
    # --mode=test   \

# test in target domain
# using char2vec_blstm
# python blstm_ner.py\
#     --data_dir=./data/ywevents/   \
#     --output_dir=./output/ywevents/char2vec_blstm/   \
#     --ckpt_model=./output/MSRA/char2vec_blstm/checkpoints/model.ckpt-31680   \
#     --vocab_file=./data/wiki/char2id.pkl   \
#     --embedding_file=./data/wiki/char2vec.txt   \
#     --mode=test   \


# test in target domain
# using blstm_crf
# python blstm_ner.py\
    # --data_dir=./data/ywevents/   \
    # --output_dir=./output/ywevents/blstm_crf/   \
    # --ckpt_model=./output/MSRA/blstm_crf/checkpoints/model.ckpt-31680   \
    # --vocab_file=./data/wiki/char2id.pkl   \
    # --embedding_file=./data/wiki/char2vec.txt   \
    # --CRF=True   \
    # --random_embedding=True   \
    # --mode=test   \


# test in target domain
# using char2vec_blstm_crf
python blstm_ner.py\
    --data_dir=./data/ywevents/   \
    --output_dir=./output/ywevents/char2vec_blstm_crf/   \
    --ckpt_model=./output/MSRA/char2vec_blstm_crf/checkpoints/model.ckpt-30888   \
    --vocab_file=./data/wiki/char2id.pkl   \
    --embedding_file=./data/wiki/char2vec.txt   \
    --CRF=True   \
    --mode=test   \