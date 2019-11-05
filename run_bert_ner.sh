#!/usr/bin/env bash

# # source domain
# python bert_ner.py\
#     --do_lower_case=False \
#     --do_train=True   \
#     --do_eval=True   \
#     --do_predict=True \
#     --data_dir=./data/ChinaDaily   \
#     --vocab_file=chinese_L-12_H-768_A-12/vocab.txt  \
#     --bert_config_file=chinese_L-12_H-768_A-12/bert_config.json \
#     --init_checkpoint=chinese_L-12_H-768_A-12/bert_model.ckpt   \
#     --max_seq_length=128   \
#     --train_batch_size=32   \
#     --learning_rate=2e-5   \
#     --num_train_epochs=4.0   \
#     --output_dir=./output/ChinaDaily/bert_crf/   \
#     --bilstm=True   \
#     --crf_only=True

# target domain
# using bert
# python bert_ner.py\
#     --do_lower_case=False \
#     --do_train=False   \
#     --do_eval=True   \
#     --do_predict=False \
#     --data_dir=./data/ywevents   \
#     --vocab_file=./chinese_L-12_H-768_A-12/vocab_update.txt  \
#     --bert_config_file=./chinese_L-12_H-768_A-12/bert_config.json \
#     --init_checkpoint=./output/ChinaDaily/bert/model.ckpt-4749   \
#     --max_seq_length=128   \
#     --train_batch_size=32   \
#     --learning_rate=2e-5   \
#     --num_train_epochs=3.0   \
#     --output_dir=./output/ywevents/bert/   \
#     --bilstm=False   \
#     --crf_only=False   \

# using bert-crf
# python bert_ner.py\
#     --do_lower_case=False \
#     --do_train=False   \
#     --do_eval=True   \
#     --do_predict=False \
#     --data_dir=./data/ywevents   \
#     --vocab_file=./chinese_L-12_H-768_A-12/vocab_update.txt  \
#     --bert_config_file=./chinese_L-12_H-768_A-12/bert_config.json \
#     --init_checkpoint=./output/ChinaDaily/bert_crf/model.ckpt-6332   \
#     --max_seq_length=128   \
#     --train_batch_size=32   \
#     --learning_rate=2e-5   \
#     --num_train_epochs=3.0   \
#     --output_dir=./output/ywevents/bert_crf/   \
#     --bilstm=True   \
#     --crf_only=True   \

# using bert-bi-lstm-crf
python bert_ner.py\
    --do_lower_case=False \
    --do_train=False   \
    --do_eval=True   \
    --do_predict=False \
    --data_dir=./data/ywevents   \
    --vocab_file=./chinese_L-12_H-768_A-12/vocab_update.txt  \
    --bert_config_file=./chinese_L-12_H-768_A-12/bert_config.json \
    --init_checkpoint=./output/ChinaDaily/bert_blstm_crf/model.ckpt-9498   \
    --max_seq_length=128   \
    --train_batch_size=32   \
    --learning_rate=2e-5   \
    --num_train_epochs=6.0   \
    --output_dir=./output/ywevents/bert_blstm_crf/   \
    --bilstm=True   \
    --crf_only=False   \
