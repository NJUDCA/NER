#! usr/bin/env python3
# -*- coding:utf-8 -*-
import numpy as np


def get_events():
    input_file = '../data/ywsz/events/time_events.txt'
    time_file = '../data/ywsz/events/time.txt'
    event_file = '../data/ywsz/events/events.txt'
    fw_t = open(time_file, "w+", encoding='UTF-8')
    fw_e = open(event_file, 'w+', encoding='UTF-8')
    with open(input_file, encoding='UTF-8') as fr:
        line = fr.readline()
        while line:
            elements = line.strip().split('\t')
            time = elements[0:-1]
            event = elements[-1]
            fw_t.write('{}\n'.format('\t'.join(str(item) for item in time)))
            fw_e.write('{}。\n'.format(event))
            line = fr.readline()
    fw_e.close()
    fw_t.close()


def max_len_events():
    event_file = '../data/ywsz/events/events.txt'
    long_event_file = '../data/ywsz/events/events_long.txt'
    fw_long = open(long_event_file, 'w+', encoding='UTF-8')

    max_seq_len = 128
    with open(event_file, encoding='UTF-8') as fr:
        events = np.loadtxt(fr, dtype=np.str)
        for i, event in enumerate(events, 1):
            if len(event) > max_seq_len:
                fw_long.write('{:05d}\t{}\t{}\n'.format(i, len(event), event))

def get_comp_unk():
    input_file = '../data/ywsz/events/predict.txt'
    predict_file = '../data/ywsz/events/label_predict.txt'
    output_file = '../data/ywsz/events/label_predict_comp.txt'
    unk_file = '../data/ywsz/events/unk.txt'
    fw_output = open(output_file, 'w+', encoding='UTF-8')
    unk_list = []

    with open(input_file, encoding='UTF-8') as fr:
        # this would skip \n row
        data = np.loadtxt(fr, dtype=np.str)
        origianl_tokens = data[:, 0]
    
    with open(predict_file, encoding='UTF-8') as fr:
        data = np.loadtxt(fr, dtype=np.str)
        predict_tokens = data[:, 0]
    
    for i, token in enumerate(origianl_tokens):
        if i < len(predict_tokens):
            predict_token = predict_tokens[i]
        else:
            predict_token = 'NULL'
        if predict_token == token:
            fw_output.write('{}\t{}\t{}\n'.format('T', token, predict_token))
        else:
            fw_output.write('{}\t{}\t{}\n'.format('F', token, predict_token))
            if predict_token == '[UNK]' and token not in unk_list:
                unk_list.append((i, token))
    
    with open(unk_file, 'w+', encoding='UTF-8') as fw_unk:
        for i, unk in unk_list:
            fw_unk.write('{}\t{}\n'.format(i, unk))
            
    fw_output.close()
    

if __name__ == "__main__":
    # get_events()
    # max_len_events()
    get_comp_unk()
