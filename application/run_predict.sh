#!/usr/bin/env bash

python predict_bert.py\
    --do_lower_case=False \
    --do_predict=True \
    --data_dir=../data/ywsz/   \
    --vocab_file=../chinese_L-12_H-768_A-12/vocab.txt  \
    --bert_config_file=../chinese_L-12_H-768_A-12/bert_config.json \
    --init_checkpoint=../output/ChinaDaily/bert_crf/model.ckpt-6332   \
    --max_seq_length=128   \
    --train_batch_size=32   \
    --learning_rate=2e-5   \
    --output_dir=../output/ChinaDaily/bert_crf/   \
    --bilstm=True   \
    --crf_only=True \
    --raw_input=True
