#!/usr/bin/env bash

# # source domain
# python bert_ner.py\
#     --do_lower_case=False \
#     --do_train=True   \
#     --do_eval=True   \
#     --do_test=True \
#     --data_dir=./data/MSRA   \
#     --vocab_file=chinese_L-12_H-768_A-12/vocab.txt  \
#     --bert_config_file=chinese_L-12_H-768_A-12/bert_config.json \
#     --init_checkpoint=chinese_L-12_H-768_A-12/bert_model.ckpt   \
#     --max_seq_length=128   \
#     --train_batch_size=32   \
#     --learning_rate=2e-5   \
#     --num_train_epochs=4.0   \
#     --output_dir=./output/MSRA/bert_crf/   \
#     --bilstm=True   \
#     --crf_only=True

# target domain
# using bert
# python bert_ner.py\
#     --do_lower_case=False \
#     --do_train=False   \
#     --do_eval=False   \
#     --do_test=True \
#     --data_dir=./data/ywevents   \
#     --vocab_file=./chinese_L-12_H-768_A-12/vocab_update.txt  \
#     --bert_config_file=./chinese_L-12_H-768_A-12/bert_config.json \
#     --init_checkpoint=./output/MSRA/bert/model.ckpt-4749   \
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
#     --do_eval=False   \
#     --do_test=True \
#     --data_dir=./data/ywevents   \
#     --vocab_file=./chinese_L-12_H-768_A-12/vocab_update.txt  \
#     --bert_config_file=./chinese_L-12_H-768_A-12/bert_config.json \
#     --init_checkpoint=./output/MSRA/bert_crf/model.ckpt-6332   \
#     --max_seq_length=128   \
#     --train_batch_size=32   \
#     --learning_rate=2e-5   \
#     --num_train_epochs=4.0   \
#     --output_dir=./output/ywevents/bert_crf/   \
#     --bilstm=True   \
#     --crf=False   \

# using bert-bi-lstm-crf
epoch=6
while [ $epoch -le 10 ]
do
    let checkpoint=`expr $epoch \* 1583`
    echo "Looping to checkpoint=$checkpoint"
    python bert_ner.py\
        --do_lower_case=False \
        --do_train=False   \
        --do_eval=True   \
        --do_test=True \
        --data_dir=./data/ywevents   \
        --vocab_file=./chinese_L-12_H-768_A-12/vocab_update.txt  \
        --bert_config_file=./chinese_L-12_H-768_A-12/bert_config.json \
        --init_checkpoint=./output/MSRA/bert_bilstm/model.ckpt-$checkpoint   \
        --max_seq_length=128   \
        --train_batch_size=32   \
        --learning_rate=2e-5   \
        --num_train_epochs=$epoch   \
        --dropout_rate=0.5   \
        --output_dir=./output/ywevents/bert_bilstm/   \
        --bilstm=True   \
        --crf=False
    ((epoch++))
done

# using bert-bi-lstm-crf
# epoch=1
# while [ $epoch -le 10 ]
# do
#     let checkpoint=`expr $epoch \* 1583`
#     echo "Looping to checkpoint=$checkpoint"
#     python bert_ner.py\
#         --do_lower_case=False \
#         --do_train=False   \
#         --do_eval=True   \
#         --do_test=True \
#         --data_dir=./data/ywevents   \
#         --vocab_file=./chinese_L-12_H-768_A-12/vocab_update.txt  \
#         --bert_config_file=./chinese_L-12_H-768_A-12/bert_config.json \
#         --init_checkpoint=./output/MSRA/bert_bilstm_crf/model.ckpt-$checkpoint   \
#         --max_seq_length=128   \
#         --train_batch_size=32   \
#         --learning_rate=2e-5   \
#         --num_train_epochs=$epoch   \
#         --dropout_rate=0.5   \
#         --output_dir=./output/ywevents/bert_bilstm_crf/   \
#         --bilstm=True   \
#         --crf=True
#     ((epoch++))
# done