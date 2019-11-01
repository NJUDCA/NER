import re
import os
import logging


class cut_max_seq:

    def __init__(self):
        self.max_seq_len = 128 - 2
        self.token_seq = []
        self.label_seq = []

    def cut(self, sent, labels):
        if len(sent) > self.max_seq_len:
            target = sent[:self.max_seq_len].rfind('；')
            if target == -1:
                target = sent[:self.max_seq_len].rfind('，')
            self.token_seq.append(sent[:target+1])
            self.label_seq.append(labels[:target+1])
            self.cut(sent[target+1:], labels[target+1:])
        else:
            self.token_seq.append(sent)
            self.label_seq.append(labels)

    def get_cut_seqs(self):
        return self.token_seq, self.label_seq


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
            sent = sent + dot
            if self.max_seq_len:
                labels = ['O'] * len(sent)
                seqs = cut_max_seq()
                seqs.cut(sent, labels)
                token_seq, _ = seqs.get_cut_seqs()
                for tokens in token_seq:
                    self.seqs.append(tokens)
            else:
                self.seqs.append(sent)
     
    def sen2seq(self, output):
        with open(output, 'w+', encoding='UTF-8') as f:
            for sent in self.seqs:
                if len(sent) > self.max_seq_len:
                    raise ValueError('input seq length {} out of max_seq_length: {}'.format(len(sent), sent))
                for char in sent:
                    if len(char.strip(' ')) > 0:
                        f.write('{} {}\n'.format(char, 'O'))
                    else:
                        continue
                f.write('\n')
