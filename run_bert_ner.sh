#!/usr/bin/env bash

python bert_ner.py\
    --do_lower_case=False \
    --do_train=True   \
    --do_eval=True   \
    --do_predict=True \
    --data_dir=./data/ChinaDaily   \
    --vocab_file=chinese_L-12_H-768_A-12/vocab.txt  \
    --bert_config_file=chinese_L-12_H-768_A-12/bert_config.json \
    --init_checkpoint=chinese_L-12_H-768_A-12/bert_model.ckpt   \
    --max_seq_length=128   \
    --train_batch_size=32   \
    --learning_rate=2e-5   \
    --num_train_epochs=6.0   \
    --output_dir=./output/ChinaDaily/bert_crf/   \
    --bilstm=True   \
    --crf_only=True
