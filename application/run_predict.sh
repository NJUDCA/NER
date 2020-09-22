#!/usr/bin/env bash

python predict_bert.py\
    --do_lower_case=False   \
    --do_predict=True   \
    --data_dir=../data/demo/   \
    --vocab_file=../chinese_L-12_H-768_A-12/vocab_update.txt  \
    --bert_config_file=../chinese_L-12_H-768_A-12/bert_config.json \
    --init_checkpoint=../output/MSRA/bert_bilstm_crf/model.ckpt-6332   \
    --max_seq_length=128   \
    --train_batch_size=32   \
    --learning_rate=2e-5   \
    --output_dir=../output/MSRA/bert_bilstm_crf/   \
    --bilstm=True   \
    --crf=True   \
    --raw_input=False   \
    --file_input=events.txt
