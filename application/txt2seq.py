import re
import os
import logging


class Txt2Seq:
    def __init__(self, data_dir, txt_file, max_seq_len=None, lang=None):
        self.lang = lang
        self.seqs = []
        self.max_seq_len = max_seq_len - 2
        input_file = '{}{}'.format(data_dir, txt_file)
        output_file = '{}predict.txt'.format(data_dir)
        if os.path.exists(input_file):
            text = open(input_file, 'r', encoding='UTF-8').read()
            sentences = text.replace('\n', '')
            self.split_sentence(sentences)
            self.sen2seq(output_file)

    def split_sentence(self, sentences):
        split_pattern = r'([.。!！?？])'
        if self.lang == 'en':
            split_pattern = r'([.?!])'
        elif self.lang == 'zn':
            split_pattern = r'([。？！])'
        sentences = re.split(split_pattern, sentences)
        for sent, dot in zip(sentences[0::2], sentences[1::2]):
            if self.max_seq_len:
                self.cut_max_seq(sent, dot)
            else:
                self.seqs.append('{}{}'.format(sent, dot))
    
    def cut_max_seq(self, seq, dot):
        if len(seq) > self.max_seq_len:
            target = seq[:self.max_seq_len].rfind('；')
            if target == -1:
                target = seq[:self.max_seq_len].rfind('，')
            self.seqs.append(seq[:target+1])
            self.cut_max_seq(seq[target+1:], dot)
        else:
            self.seqs.append('{}{}'.format(seq, dot))
     
    def sen2seq(self, output):
        with open(output, 'w+', encoding='UTF-8') as f:
            for sent in self.seqs:
                if len(sent) > self.max_seq_len:
                    raise ValueError('input seq length out of max_seq_length: {}'.format(sent))
                for char in sent:
                    if len(char.strip(' ')) > 0:
                        f.write('{} {}\n'.format(char, 'O'))
                    else:
                        continue
                f.write('\n')
